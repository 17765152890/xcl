//! Virtual file system (design §5.4).
//!
//! * Console path prefixes (`G:\`, `GAME:`, `T:\` …) are translated via mounts.
//! * The game folder is **read-only**. A write to a game file first copies that single
//!   file into `<data>\overlay\` and the write goes to the copy. Reads prefer the
//!   overlay. Deleting a game file records a "whiteout" instead of touching the original.
//! * This module is pure logic + std::fs; the Win32 hooks that feed it live in
//!   `vfs_hooks.rs`.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// Not ours – use the path unchanged.
    Pass,
    /// A console path translated to a real, writable location (e.g. T:\ → data\temp).
    Translated(String),
    /// Inside the (read-only) game folder; `rel` has no leading separator ("" = root).
    Game { rel: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Pass,
    Use(String),
    /// Fail with this Win32 error code.
    Fail(u32),
}

pub const ERROR_FILE_NOT_FOUND: u32 = 2;
pub const ERROR_PATH_NOT_FOUND: u32 = 3;
pub const ERROR_ACCESS_DENIED: u32 = 5;
pub const ERROR_FILE_EXISTS: u32 = 80;
pub const ERROR_ALREADY_EXISTS: u32 = 183;
pub const ERROR_DIR_NOT_EMPTY: u32 = 145;

const CREATE_NEW: u32 = 1;
const CREATE_ALWAYS: u32 = 2;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const TRUNCATE_EXISTING: u32 = 5;

const WRITE_ACCESS: u32 = 0x4000_0000 // GENERIC_WRITE
    | 0x1000_0000 // GENERIC_ALL
    | 0x0000_0002 // FILE_WRITE_DATA / ADD_FILE
    | 0x0000_0004 // FILE_APPEND_DATA / ADD_SUBDIRECTORY
    | 0x0000_0010 // FILE_WRITE_EA
    | 0x0000_0100 // FILE_WRITE_ATTRIBUTES
    | 0x0001_0000 // DELETE
    | 0x0004_0000 // WRITE_DAC
    | 0x0008_0000; // WRITE_OWNER
const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;

pub struct Vfs {
    game_root: String,
    game_root_lc: String,
    /// the XCL data folder (host territory: TLS carriers, caches, logs)
    data_root_lc: String,
    overlay: String,
    mounts: Vec<(String, String)>,
    whiteouts: Mutex<HashSet<String>>,
    whiteout_file: PathBuf,
    existing_drives: u32,
    unmapped: Mutex<BTreeSet<String>>,
    writes: Mutex<BTreeSet<String>>,
}

static VFS: OnceLock<Vfs> = OnceLock::new();

pub fn init(v: Vfs) -> &'static Vfs {
    let _ = VFS.set(v);
    VFS.get().unwrap()
}

pub fn get() -> Option<&'static Vfs> {
    VFS.get()
}

fn norm(s: &str) -> String {
    s.replace('/', "\\")
}

fn trim_sep(s: &str) -> String {
    s.trim_end_matches('\\').to_string()
}

fn strip_verbatim(p: &str) -> String {
    if let Some(r) = p.strip_prefix("\\\\?\\UNC\\") {
        return format!("\\\\{r}");
    }
    if let Some(r) = p.strip_prefix("\\\\?\\") {
        return r.to_string();
    }
    if let Some(r) = p.strip_prefix("\\??\\") {
        return r.to_string();
    }
    p.to_string()
}

fn is_device(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    if l.starts_with("\\\\.\\") || l.starts_with("\\\\?\\globalroot") {
        return true;
    }
    let base = l.trim_end_matches(':');
    matches!(base, "con" | "nul" | "prn" | "aux" | "conin$" | "conout$" | "clock$")
        || (base.len() == 4 && (base.starts_with("com") || base.starts_with("lpt")) && base.as_bytes()[3].is_ascii_digit())
}

/// Full path via GetFullPathNameW (no file system access).
pub fn full_path(p: &str) -> Option<String> {
    use windows_sys::Win32::Storage::FileSystem::GetFullPathNameW;
    let w = crate::util::wide(p);
    let mut buf = vec![0u16; 1024];
    let n = unsafe { GetFullPathNameW(w.as_ptr(), buf.len() as u32, buf.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    if n == 0 {
        return None;
    }
    if n > buf.len() {
        buf.resize(n + 1, 0);
        let n2 = unsafe { GetFullPathNameW(w.as_ptr(), buf.len() as u32, buf.as_mut_ptr(), std::ptr::null_mut()) } as usize;
        return (n2 > 0).then(|| String::from_utf16_lossy(&buf[..n2]));
    }
    Some(String::from_utf16_lossy(&buf[..n]))
}

fn logical_drives() -> u32 {
    unsafe { windows_sys::Win32::Storage::FileSystem::GetLogicalDrives() }
}

impl Vfs {
    pub fn new(game_root: &Path, data_dir: &Path, mounts: Vec<(String, String)>) -> Vfs {
        let game_root = trim_sep(&norm(&game_root.to_string_lossy()));
        let data_root = trim_sep(&norm(&data_dir.to_string_lossy()));
        let overlay_dir = data_dir.join("overlay");
        let whiteout_file = overlay_dir.join(".xcl_whiteouts");
        let whiteouts = std::fs::read_to_string(&whiteout_file)
            .map(|s| s.lines().filter(|l| !l.is_empty()).map(str::to_string).collect())
            .unwrap_or_default();
        let mut mounts: Vec<(String, String)> = mounts
            .into_iter()
            .map(|(p, t)| (p.to_ascii_lowercase().trim_end_matches(['\\', '/']).to_string(), trim_sep(&norm(&t))))
            .collect();
        // longest prefix first ("GAME:" before "G:")
        mounts.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Vfs {
            game_root_lc: game_root.to_lowercase(),
            game_root,
            data_root_lc: data_root.to_lowercase(),
            overlay: trim_sep(&norm(&overlay_dir.to_string_lossy())),
            mounts,
            whiteouts: Mutex::new(whiteouts),
            whiteout_file,
            existing_drives: logical_drives(),
            unmapped: Mutex::new(BTreeSet::new()),
            writes: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn game_root(&self) -> &str {
        &self.game_root
    }

    /// Whether the lowercased path `l` is `root` itself or under it.
    fn under(l: &str, root: &str) -> bool {
        l == root || (l.starts_with(root) && l.as_bytes().get(root.len()) == Some(&b'\\'))
    }

    fn apply_mounts(&self, p: &str) -> Option<String> {
        // to_ascii_lowercase keeps byte offsets, so slicing `p` with prefix.len() is safe
        let l = p.to_ascii_lowercase();
        for (prefix, target) in &self.mounts {
            if !l.starts_with(prefix.as_str()) {
                continue;
            }
            // A path already under a host root - the mount target (the game folder) or
            // the XCL data folder - is a real host path, not a console one: the default
            // configuration mounts a drive letter to the game folder ("D:" = "{game}"),
            // and a workspace on that drive has its own paths there (the game exe, the
            // TLS carriers, the shader cache) which would otherwise translate to
            // doubled, nonexistent paths. Console paths never include the host-side
            // game folder path (the game folder IS the drive root for them).
            let tl = target.to_ascii_lowercase();
            if Self::under(&l, &tl) || Self::under(&l, &self.data_root_lc) {
                continue;
            }
            let rest = &p[prefix.len()..];
            // "G:" / "GAME:" accept anything after the colon ("G:foo" == "G:\foo");
            // directory-style prefixes need a separator boundary.
            if !(prefix.ends_with(':') || rest.is_empty() || rest.starts_with('\\')) {
                continue;
            }
            let tail = rest.trim_start_matches('\\');
            return Some(if tail.is_empty() { target.clone() } else { format!("{target}\\{tail}") });
        }
        // Embedded "...\GAME:\rest": some runtimes treat "GAME:\x" as a relative path and
        // prepend the current directory. A component ending in ':' is never a valid file
        // name, so it can only be a console mount.
        for (prefix, target) in &self.mounts {
            if !prefix.ends_with(':') {
                continue;
            }
            let needle = format!("\\{prefix}");
            let mut from = 0;
            while let Some(i) = l[from..].find(&needle) {
                let at = from + i;
                let after = at + needle.len();
                let rest = &p[after..];
                if rest.is_empty() || rest.starts_with('\\') {
                    let tail = rest.trim_start_matches('\\');
                    return Some(if tail.is_empty() { target.clone() } else { format!("{target}\\{tail}") });
                }
                from = after;
            }
        }
        None
    }

    /// Classify a path using `full` to absolutise (injected for unit tests).
    pub fn classify_with(&self, path: &str, full: impl Fn(&str) -> Option<String>) -> Target {
        if path.is_empty() {
            return Target::Pass;
        }
        let p = norm(path);
        if is_device(&p) {
            return Target::Pass;
        }
        let p = strip_verbatim(&p);
        if p.starts_with("\\\\") {
            return Target::Pass; // UNC
        }
        let (p, translated) = match self.apply_mounts(&p) {
            Some(t) => (t, true),
            None => (p, false),
        };
        if !translated {
            let b = p.as_bytes();
            if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
                let bit = (b[0].to_ascii_uppercase() - b'A') as u32;
                if self.existing_drives & (1 << bit) == 0 {
                    let mut u = self.unmapped.lock().unwrap();
                    if u.len() < 1000 && u.insert(p.clone()) {
                        crate::xlog_warn!("[VFS] unmapped console path: {p}");
                    }
                    return Target::Pass;
                }
            }
        }
        let Some(abs) = full(&p) else { return Target::Pass };
        let abs = trim_sep(&strip_verbatim(&norm(&abs)));
        if let Some(t) = self.game_target(&abs) {
            return t;
        }
        if translated {
            Target::Translated(abs)
        } else {
            Target::Pass
        }
    }

    /// Game target of an absolute path: the root itself, `rel` when under it (None otherwise).
    fn game_target(&self, abs: &str) -> Option<Target> {
        let lc = abs.to_lowercase();
        if lc == self.game_root_lc {
            return Some(Target::Game { rel: String::new() });
        }
        let rest = lc.strip_prefix(&self.game_root_lc)?;
        if !rest.starts_with('\\') {
            return None;
        }
        Some(Target::Game { rel: abs[self.game_root.len() + 1..].to_string() })
    }

    pub fn classify(&self, path: &str) -> Target {
        self.classify_with(path, full_path)
    }

    pub fn is_game_path(&self, p: &Path) -> bool {
        matches!(self.classify(&p.to_string_lossy()), Target::Game { .. })
    }

    pub fn overlay_path(&self, rel: &str) -> String {
        if rel.is_empty() {
            self.overlay.clone()
        } else {
            format!("{}\\{rel}", self.overlay)
        }
    }

    pub fn original_path(&self, rel: &str) -> String {
        if rel.is_empty() {
            self.game_root.clone()
        } else {
            format!("{}\\{rel}", self.game_root)
        }
    }

    fn wkey(rel: &str) -> String {
        rel.to_lowercase()
    }

    pub fn is_whiteout(&self, rel: &str) -> bool {
        let k = Self::wkey(rel);
        let g = self.whiteouts.lock().unwrap();
        if g.is_empty() {
            return false;
        }
        // a whiteout on a parent directory hides everything below it
        if g.contains(&k) {
            return true;
        }
        let mut cur = k.as_str();
        while let Some((parent, _)) = cur.rsplit_once('\\') {
            if g.contains(parent) {
                return true;
            }
            cur = parent;
        }
        false
    }

    pub fn has_whiteouts_under(&self, rel_dir: &str) -> bool {
        let k = Self::wkey(rel_dir);
        let g = self.whiteouts.lock().unwrap();
        if k.is_empty() {
            return !g.is_empty();
        }
        let pre = format!("{k}\\");
        g.iter().any(|w| w.starts_with(&pre))
    }

    fn persist_whiteouts(&self, g: &HashSet<String>) {
        let _ = std::fs::create_dir_all(&self.overlay);
        let mut v: Vec<&String> = g.iter().collect();
        v.sort();
        let body: String = v.into_iter().map(|s| format!("{s}\n")).collect();
        let _ = std::fs::write(&self.whiteout_file, body);
    }

    fn add_whiteout(&self, rel: &str) {
        let mut g = self.whiteouts.lock().unwrap();
        if g.insert(Self::wkey(rel)) {
            self.persist_whiteouts(&g);
        }
    }

    fn remove_whiteout(&self, rel: &str) {
        let mut g = self.whiteouts.lock().unwrap();
        let k = Self::wkey(rel);
        let mut changed = g.remove(&k);
        // re-creating something below a whited-out parent un-hides the parent path
        let mut cur = k.as_str();
        while let Some((parent, _)) = cur.rsplit_once('\\') {
            changed |= g.remove(parent);
            cur = parent;
        }
        if changed {
            self.persist_whiteouts(&g);
        }
    }

    fn record_write(&self, rel: &str) {
        let mut w = self.writes.lock().unwrap();
        if w.len() < 10_000 && w.insert(rel.to_string()) {
            crate::xlog_info!("[VFS] write to a game file redirected to the overlay: {rel}");
        }
    }

    pub fn visible_in_overlay(&self, rel: &str) -> bool {
        Path::new(&self.overlay_path(rel)).exists()
    }

    pub fn visible_in_original(&self, rel: &str) -> bool {
        !self.is_whiteout(rel) && Path::new(&self.original_path(rel)).exists()
    }

    /// Real path to read `path` from. None = hidden by a whiteout.
    pub fn real_path_for_read(&self, path: &str) -> Option<String> {
        match self.classify(path) {
            Target::Pass => Some(path.to_string()),
            Target::Translated(p) => Some(p),
            Target::Game { rel } => self.read_path_rel(&rel),
        }
    }

    pub fn read_path_rel(&self, rel: &str) -> Option<String> {
        if self.visible_in_overlay(rel) {
            return Some(self.overlay_path(rel));
        }
        if self.is_whiteout(rel) {
            return None;
        }
        Some(self.original_path(rel))
    }

    /// Ensure `rel` exists in the overlay (copying the original when `keep_content`)
    /// and return the overlay path.
    pub fn prepare_write(&self, rel: &str, keep_content: bool) -> Result<String, u32> {
        let ov = self.overlay_path(rel);
        if let Some(parent) = Path::new(&ov).parent() {
            std::fs::create_dir_all(parent).map_err(|_| ERROR_PATH_NOT_FOUND)?;
        }
        if keep_content && !Path::new(&ov).exists() && self.visible_in_original(rel) {
            let orig = self.original_path(rel);
            if Path::new(&orig).is_dir() {
                std::fs::create_dir_all(&ov).map_err(|_| ERROR_ACCESS_DENIED)?;
            } else {
                std::fs::copy(&orig, &ov).map_err(|_| ERROR_ACCESS_DENIED)?;
            }
        }
        self.remove_whiteout(rel);
        self.record_write(rel);
        Ok(ov)
    }

    fn is_write(access: u32, disposition: u32, flags: u32) -> bool {
        access & WRITE_ACCESS != 0
            || matches!(disposition, CREATE_NEW | CREATE_ALWAYS | TRUNCATE_EXISTING)
            || flags & FILE_FLAG_DELETE_ON_CLOSE != 0
    }

    pub fn open_decision_for(&self, target: Target, access: u32, disposition: u32, flags: u32) -> Decision {
        match target {
            Target::Pass => Decision::Pass,
            Target::Translated(p) => {
                if Self::is_write(access, disposition, flags) || disposition == OPEN_ALWAYS {
                    if let Some(parent) = Path::new(&p).parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
                Decision::Use(p)
            }
            Target::Game { rel } => {
                let in_ov = self.visible_in_overlay(&rel);
                let in_orig = self.visible_in_original(&rel);
                let visible = in_ov || in_orig;
                let write = Self::is_write(access, disposition, flags) || (disposition == OPEN_ALWAYS && !visible);
                if !write {
                    return match self.read_path_rel(&rel) {
                        Some(p) => Decision::Use(p),
                        None => Decision::Fail(ERROR_FILE_NOT_FOUND),
                    };
                }
                match disposition {
                    CREATE_NEW if visible => return Decision::Fail(ERROR_FILE_EXISTS),
                    OPEN_EXISTING | TRUNCATE_EXISTING if !visible => return Decision::Fail(ERROR_FILE_NOT_FOUND),
                    _ => {}
                }
                let keep = matches!(disposition, OPEN_EXISTING | OPEN_ALWAYS);
                match self.prepare_write(&rel, keep) {
                    Ok(p) => Decision::Use(p),
                    Err(e) => Decision::Fail(e),
                }
            }
        }
    }

    pub fn open_decision(&self, path: &str, access: u32, disposition: u32, flags: u32) -> Decision {
        self.open_decision_for(self.classify(path), access, disposition, flags)
    }

    pub fn delete_file(&self, rel: &str) -> Result<(), u32> {
        let ov = self.overlay_path(rel);
        let had_ov = Path::new(&ov).is_file();
        let had_orig = self.visible_in_original(rel);
        if !had_ov && !had_orig {
            return Err(ERROR_FILE_NOT_FOUND);
        }
        if had_ov {
            std::fs::remove_file(&ov).map_err(|_| ERROR_ACCESS_DENIED)?;
        }
        if had_orig {
            self.add_whiteout(rel);
        }
        self.record_write(rel);
        Ok(())
    }

    pub fn create_dir(&self, rel: &str) -> Result<(), u32> {
        if self.visible_in_overlay(rel) || self.visible_in_original(rel) {
            return Err(ERROR_ALREADY_EXISTS);
        }
        std::fs::create_dir_all(self.overlay_path(rel)).map_err(|_| ERROR_PATH_NOT_FOUND)?;
        self.remove_whiteout(rel);
        self.record_write(rel);
        Ok(())
    }

    pub fn remove_dir(&self, rel: &str) -> Result<(), u32> {
        let ov = self.overlay_path(rel);
        let had_ov = Path::new(&ov).is_dir();
        let had_orig = self.visible_in_original(rel);
        if !had_ov && !had_orig {
            return Err(ERROR_FILE_NOT_FOUND);
        }
        if had_ov {
            std::fs::remove_dir(&ov).map_err(|_| ERROR_DIR_NOT_EMPTY)?;
        }
        if had_orig {
            self.add_whiteout(rel);
        }
        self.record_write(rel);
        Ok(())
    }

    /// Whether directory listings under `rel_dir` must merge overlay + original.
    pub fn needs_merge(&self, rel_dir: &str) -> bool {
        Path::new(&self.overlay_path(rel_dir)).is_dir() || self.has_whiteouts_under(rel_dir)
    }

    pub fn unmapped_paths(&self) -> Vec<String> {
        self.unmapped.lock().map(|g| g.iter().cloned().collect()).unwrap_or_default()
    }

    pub fn redirected_writes(&self) -> Vec<String> {
        self.writes.lock().map(|g| g.iter().cloned().collect()).unwrap_or_default()
    }

    pub fn whiteouts(&self) -> Vec<String> {
        let mut v: Vec<String> = self.whiteouts.lock().map(|g| g.iter().cloned().collect()).unwrap_or_default();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        dir: PathBuf,
        vfs: Vfs,
    }

    impl Fixture {
        fn new(tag: &str) -> Fixture {
            let dir = std::env::temp_dir().join(format!("xcl_vfs_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let game = dir.join("game");
            std::fs::create_dir_all(game.join("data")).unwrap();
            std::fs::write(game.join("data").join("hello.txt"), b"original").unwrap();
            std::fs::write(game.join("data").join("gone.txt"), b"bye").unwrap();
            let data = dir.join("xcldata");
            let mounts = vec![
                ("G:".to_string(), game.to_string_lossy().to_string()),
                ("GAME:".to_string(), game.to_string_lossy().to_string()),
                ("T:".to_string(), data.join("temp").to_string_lossy().to_string()),
            ];
            let vfs = Vfs::new(&game, &data, mounts);
            Fixture { dir, vfs }
        }
        fn game(&self) -> PathBuf {
            self.dir.join("game")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn mounts_and_classification() {
        let f = Fixture::new("cls");
        let v = &f.vfs;
        assert_eq!(v.classify("G:\\data\\hello.txt"), Target::Game { rel: "data\\hello.txt".into() });
        assert_eq!(v.classify("g:/data/hello.txt"), Target::Game { rel: "data\\hello.txt".into() });
        assert_eq!(v.classify("GAME:\\data\\hello.txt"), Target::Game { rel: "data\\hello.txt".into() });
        assert_eq!(v.classify("G:\\"), Target::Game { rel: String::new() });
        // "GAME:\x" absolutised by a runtime as if it were relative
        let mangled = format!("\\\\?\\{}\\GAME:\\data\\hello.txt", f.game().display());
        assert_eq!(v.classify(&mangled), Target::Game { rel: "data\\hello.txt".into() });
        assert_eq!(v.classify("C:\\somewhere\\G:\\data\\hello.txt"), Target::Game { rel: "data\\hello.txt".into() });
        let real = f.game().join("data").join("hello.txt");
        assert_eq!(v.classify(&real.to_string_lossy()), Target::Game { rel: "data\\hello.txt".into() });
        assert!(matches!(v.classify("T:\\scratch.bin"), Target::Translated(p) if p.ends_with("xcldata\\temp\\scratch.bin")));
        assert_eq!(v.classify("CONOUT$"), Target::Pass);
        assert_eq!(v.classify("\\\\.\\pipe\\x"), Target::Pass);
        assert_eq!(v.classify("C:\\Windows\\notepad.exe"), Target::Pass);
    }

    #[test]
    fn a_drive_letter_mount_never_hides_real_game_paths() {
        // The default title configuration mounts a drive letter to the game folder
        // ("D:" = "{game}"); when the workspace itself is on that drive, the host's own
        // real paths under the game root (the game exe, for one) must stay game paths
        // instead of being mount-translated to doubled, nonexistent paths.
        let f = Fixture::new("mnt");
        let game = f.game();
        let drive = game
            .components()
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy()
            .trim_end_matches('\\')
            .to_string();
        let vfs = Vfs::new(&game, &f.dir.join("data2"), vec![(drive.clone(), game.to_string_lossy().into_owned())]);
        let exe = game.join("the_title.exe");
        assert_eq!(vfs.classify(&exe.to_string_lossy()), Target::Game { rel: "the_title.exe".into() });
        // a host path under the XCL data folder (TLS carriers, shader caches) stays real
        let carrier = f.dir.join("data2").join("cache").join("xcl_tls_carrier.dll");
        assert!(matches!(vfs.classify(&carrier.to_string_lossy()), Target::Pass));
        // the mount still maps the rest of that drive into the game folder
        assert_eq!(vfs.classify(&format!("{drive}\\elsewhere\\x.bin")), Target::Game { rel: "elsewhere\\x.bin".into() });
    }

    #[test]
    fn reads_never_touch_and_writes_copy_on_write() {
        let f = Fixture::new("cow");
        let v = &f.vfs;
        let orig = f.game().join("data").join("hello.txt");
        // read → original
        assert_eq!(
            v.open_decision("G:\\data\\hello.txt", 0x8000_0000, OPEN_EXISTING, 0),
            Decision::Use(orig.to_string_lossy().to_string())
        );
        // write → overlay copy with original content
        let d = v.open_decision("G:\\data\\hello.txt", 0x4000_0000, OPEN_EXISTING, 0);
        let Decision::Use(ov) = d else { panic!("{d:?}") };
        assert!(ov.contains("overlay"));
        assert_eq!(std::fs::read(&ov).unwrap(), b"original");
        std::fs::write(&ov, b"modified").unwrap();
        // later reads see the overlay
        assert_eq!(v.open_decision("G:\\data\\hello.txt", 0x8000_0000, OPEN_EXISTING, 0), Decision::Use(ov.clone()));
        // original untouched
        assert_eq!(std::fs::read(&orig).unwrap(), b"original");
        // CREATE_ALWAYS on a new file → overlay, nothing created in the game folder
        let Decision::Use(newp) = v.open_decision("G:\\data\\new.bin", 0x4000_0000, CREATE_ALWAYS, 0) else { panic!() };
        assert!(newp.contains("overlay"));
        assert!(!f.game().join("data").join("new.bin").exists());
        // CREATE_NEW on an existing file fails
        assert_eq!(v.open_decision("G:\\data\\hello.txt", 0x4000_0000, CREATE_NEW, 0), Decision::Fail(ERROR_FILE_EXISTS));
        assert_eq!(v.redirected_writes(), vec!["data\\hello.txt".to_string(), "data\\new.bin".to_string()]);
    }

    #[test]
    fn delete_is_a_persistent_whiteout() {
        let f = Fixture::new("wo");
        let v = &f.vfs;
        assert_eq!(v.delete_file("data\\gone.txt"), Ok(()));
        assert!(f.game().join("data").join("gone.txt").exists(), "original must stay");
        assert_eq!(v.open_decision("G:\\data\\gone.txt", 0x8000_0000, OPEN_EXISTING, 0), Decision::Fail(ERROR_FILE_NOT_FOUND));
        assert_eq!(v.delete_file("data\\gone.txt"), Err(ERROR_FILE_NOT_FOUND));
        assert!(v.needs_merge("data"));
        // persisted
        let v2 = Vfs::new(&f.game(), &f.dir.join("xcldata"), vec![]);
        assert!(v2.is_whiteout("data\\gone.txt"));
        // re-creating the file un-hides it (in the overlay)
        let Decision::Use(p) = v.open_decision("G:\\data\\gone.txt", 0x4000_0000, CREATE_ALWAYS, 0) else { panic!() };
        assert!(p.contains("overlay"));
        assert!(!v.is_whiteout("data\\gone.txt"));
    }

    #[test]
    fn directories() {
        let f = Fixture::new("dir");
        let v = &f.vfs;
        assert_eq!(v.create_dir("data"), Err(ERROR_ALREADY_EXISTS));
        assert_eq!(v.create_dir("saves\\slot1"), Ok(()));
        assert!(!f.game().join("saves").exists());
        assert!(v.visible_in_overlay("saves\\slot1"));
        assert_eq!(v.remove_dir("data"), Ok(()));
        assert!(f.game().join("data").exists());
        assert!(v.is_whiteout("data\\hello.txt"), "children of a removed dir are hidden");
    }
}
