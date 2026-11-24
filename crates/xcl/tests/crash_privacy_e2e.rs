//! `cargo build -p xcl-testtitle --bin xcl_privacy_probe` first. All traffic stays on loopback.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn run_launcher(command: &mut Command, output: &Path) -> std::io::Result<Option<ExitStatus>> {
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::JobObjects::*;
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    // Files, not inherited pipes: collecting diagnostics cannot wait on the guest closing stdout.
    let capture = std::fs::File::create(output)?;
    command.stdout(Stdio::from(capture.try_clone()?)).stderr(Stdio::from(capture));
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = OwnedHandle::from_raw_handle(job);
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            job.as_raw_handle(), JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), std::mem::size_of_val(&limits) as u32,
        ) == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut child = command.spawn()?;
        // Closing the job also terminates the title host if it gets stuck behind the launcher.
        if AssignProcessToJobObject(job.as_raw_handle(), child.as_raw_handle()) == 0 {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            return Err(error);
        }
        match WaitForSingleObject(child.as_raw_handle(), 30_000) {
            WAIT_OBJECT_0 => child.try_wait(),
            WAIT_TIMEOUT => Ok(None),
            _ => Err(std::io::Error::last_os_error()),
        }
    }
}

fn time_left(deadline: Instant) -> io::Result<Duration> {
    deadline.checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| io::ErrorKind::TimedOut.into())
}

// Apply one deadline to the underlying I/O, including partial reads and writes.
struct ReportStream {
    stream: TcpStream,
    deadline: Instant,
}

impl Read for ReportStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream.set_read_timeout(Some(time_left(self.deadline)?))?;
        self.stream.read(buf)
    }
}

impl Write for ReportStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.set_write_timeout(Some(time_left(self.deadline)?))?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

fn serve_report(listener: &TcpListener, expected: &str, deadline: Instant, launcher_done: mpsc::Receiver<()>) {
    let stream = loop {
        let remaining = time_left(deadline).expect("timed out accepting the loopback POST");
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                let poll = remaining.min(Duration::from_millis(10));
                assert_eq!(launcher_done.recv_timeout(poll), Err(mpsc::RecvTimeoutError::Timeout),
                    "launcher exited before a POST reached the loopback listener");
            }
            Err(error) => panic!("could not accept the loopback POST: {error}"),
        }
    };
    stream.set_nonblocking(false).unwrap();
    let mut request = BufReader::new(ReportStream { stream, deadline });
    let mut line = String::new();
    request.read_line(&mut line).unwrap();
    assert!(line.starts_with("POST /synthetic-report "), "{line:?}");
    let mut len = None;
    loop {
        line.clear();
        assert_ne!(request.read_line(&mut line).unwrap(), 0, "incomplete HTTP headers");
        if line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("Content-Length") {
                len = Some(value.trim().parse::<usize>().unwrap());
            }
        }
    }
    assert_eq!(len, Some(expected.len()));
    let mut body = vec![0; expected.len()];
    request.read_exact(&mut body).unwrap();
    assert_eq!(body, expected.as_bytes());
    request.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
}

#[test]
fn title_crash_report_environment_and_offline_winhttp() {
    let xcl = Path::new(env!("CARGO_BIN_EXE_xcl"));
    let probe = xcl.with_file_name("xcl_privacy_probe.exe");
    assert!(probe.is_file(), "build the privacy fixture first: cargo build -p xcl-testtitle --bin xcl_privacy_probe");
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = std::env::temp_dir().join(format!("xcl-privacy-{}-{nonce}", std::process::id()));
    let game = root.join("game");
    let data = root.join("data-雪");
    let profile = root.join("profile-雪");
    std::fs::create_dir_all(&game).unwrap();
    std::fs::create_dir_all(&profile).unwrap();
    std::fs::create_dir_all(data.join("game")).unwrap();
    std::fs::copy(probe, game.join("xcl_privacy_probe.exe")).unwrap();
    // Use native WinHTTP even offline: the #305 system hooks must work without an HLE IAT override.
    std::fs::write(data.join("game").join("xcl.toml"), "graphics = 'null'\n[aliases]\n'winhttp.dll' = '@system'\n").unwrap();

    const DUMMY: &str = "test-only=雪";
    for (inherit, offline, expected) in [
        (false, false, Some("synthetic-report:".to_owned())),
        (true, false, Some(format!("synthetic-report:{DUMMY}"))),
        (false, true, None),
    ] {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut command = Command::new(xcl);
        command
            .arg("run").arg(&game).arg("--").arg(listener.local_addr().unwrap().port().to_string())
            .env_clear() // Never even put the test runner's credentials in the launcher process.
            .env("SystemRoot", std::env::var_os("SystemRoot").unwrap())
            .env("TEMP", &profile)
            .env("localappdata", &profile)
            .env("XCL_DATA_ROOT", &data)
            .env("XCL_SKIP_CPU_CHECK", "1")
            .env("xcl_privacy_marker", "keep=雪")
            .env("DUMMY_CRASH_SECRET", DUMMY);
        if inherit {
            command.env("XCL_INHERIT_ENV", "1");
        }
        if offline {
            command.env("xcl_network", "0");
        }
        let output = root.join("xcl-output.txt");
        let (result, report) = if let Some(expected) = expected {
            thread::scope(|scope| {
                // The probe waits for the response before exiting.
                let deadline = Instant::now() + Duration::from_secs(25);
                let (accept_cancel, launcher_done) = mpsc::channel();
                let listener = &listener;
                let worker = scope.spawn(move || serve_report(listener, &expected, deadline, launcher_done));
                let result = run_launcher(&mut command, &output);
                drop(accept_cancel);
                (result, Some(worker.join()))
            })
        } else {
            (run_launcher(&mut command, &output), None)
        };
        let diagnostics = std::fs::read_to_string(output).unwrap_or_default();
        let log = std::fs::read_to_string(data.join("game").join("logs").join("xcl.log")).unwrap_or_default();
        if let Some(Err(panic)) = report {
            eprintln!("loopback POST failed (inherit={inherit}, launcher={result:?})\n{diagnostics}\n{log}");
            std::panic::resume_unwind(panic);
        }
        let status = result.unwrap_or_else(|error| panic!("could not run xcl: {error}\n{diagnostics}\n{log}"))
            .unwrap_or_else(|| panic!("xcl did not exit within 30 s (inherit={inherit}, offline={offline})\n{diagnostics}\n{log}"));
        let expected_status = if offline { 12007 } else { 0 }; // ERROR_WINHTTP_NAME_NOT_RESOLVED
        assert_eq!(status.code(), Some(expected_status), "inherit={inherit}, offline={offline}\n{diagnostics}\n{log}");
        if offline {
            assert!(log.contains("WinHttpConnect fails (no network)"), "{log}");
            assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
