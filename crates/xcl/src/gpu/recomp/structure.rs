//! Structured control flow for any reducible program. `structurable` works on the layout:
//! every backward jump is a loop to it, every forward one a skip. Code the console compiler
//! placed after the blocks it jumps back into, entered from before them (acyclic, though its
//! jumps go backwards), and loops with several heads or exits it cannot place made programs
//! fall back to the `pc` dispatcher (`emit_range`). Those
//! programs were 12% of the retail pack's but about two thirds of its DXC time: every value
//! of a dispatcher loop is loop-carried, and DXC's jump threading and loop passes work on it.
//!
//! Method (N. Ramsey, "Beyond Relooper", ICFP 2022): over the dominator tree, a loop head
//! becomes `for (;;)`, a block with several forward predecessors (a merge) is reached by
//! leaving a `do { … } while (false)` that ends right before its code, any other block is
//! emitted inline at its only predecessor's branch. A loop's exits that it dominates are
//! placed after the loop. A jump that leaves several of these constructs at once sets
//! `xcl_br` to its target and each construct passes it outwards.
//! `XCL_RECOMP_GENERAL=1` uses this form for every program with branches (tests).

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::{block_end, emit, insts_in, term, Ctx, Term, LOOP_GUARD};

/// Successor of a block: another block, or the end of the program.
type Target = Option<usize>;

#[derive(Clone, Copy, Debug)]
enum Succ {
    Exit,
    Jump(usize),
    /// (taken, not taken)
    Cond(Target, Target),
}

/// Basic blocks of the program (the leaders of `Ctx`) and their successors.
struct Cfg {
    succ: Vec<Succ>,
}

fn cfg(c: &Ctx) -> Result<Cfg, String> {
    let n = c.leaders.len();
    let index = |t: u64| c.leaders.binary_search(&t).map_err(|_| format!("branch to {t:#x}: not a block"));
    let mut succ = Vec::with_capacity(n);
    for i in 0..n {
        let (start, end) = (c.leaders[i], block_end(c, i));
        let fall: Target = if i + 1 < n && c.leaders[i + 1] == end { Some(i + 1) } else { None };
        let s = match term(c, start, end) {
            Term::Next => match fall {
                Some(f) => Succ::Jump(f),
                None => Succ::Exit,
            },
            Term::Jump(t) => Succ::Jump(index(t)?),
            Term::Cond(t) => {
                let t = index(t)?;
                if Some(t) == fall {
                    Succ::Jump(t)
                } else {
                    Succ::Cond(Some(t), fall)
                }
            }
            Term::End => Succ::Exit,
            Term::Other => return Err(format!("block {start:#x}: a terminator without a static target")),
        };
        succ.push(s);
    }
    Ok(Cfg { succ })
}

impl Succ {
    fn targets(self) -> Vec<usize> {
        match self {
            Succ::Exit => vec![],
            Succ::Jump(t) => vec![t],
            Succ::Cond(t, f) => t.into_iter().chain(f).collect(),
        }
    }
}

/// Dominator tree, loops and merges of the reachable blocks.
struct Shape {
    /// reverse postorder number (usize::MAX: unreachable)
    rpo: Vec<usize>,
    order: Vec<usize>,
    /// dominator tree children in reverse postorder
    children: Vec<Vec<usize>>,
    /// loop head → its body (blocks), heads only
    body: Vec<Option<Vec<bool>>>,
    /// forward edges into the block
    forward_in: Vec<usize>,
    /// the loop head after whose loop the block is placed (it is outside the loops of its
    /// immediate dominator that it leaves; the outermost of them)
    placed_after: Vec<Option<usize>>,
}

fn shape(g: &Cfg) -> Result<Shape, String> {
    let n = g.succ.len();
    // reverse postorder (iterative DFS from block 0)
    let mut post = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        let ts = g.succ[b].targets();
        if *k < ts.len() {
            let t = ts[*k];
            *k += 1;
            if !seen[t] {
                seen[t] = true;
                stack.push((t, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    let order: Vec<usize> = post.into_iter().rev().collect();
    let mut rpo = vec![usize::MAX; n];
    for (k, &b) in order.iter().enumerate() {
        rpo[b] = k;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in &order {
        for t in g.succ[b].targets() {
            preds[t].push(b);
        }
    }
    // dominators (Cooper, Harvey, Kennedy)
    let mut idom = vec![usize::MAX; n];
    idom[0] = 0;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while rpo[a] > rpo[b] {
                a = idom[a];
            }
            while rpo[b] > rpo[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut new = usize::MAX;
            for &p in &preds[b] {
                if idom[p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX { p } else { intersect(&idom, p, new) };
            }
            if new != idom[b] {
                idom[b] = new;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == 0 {
            return false;
        }
        b = idom[b];
    };
    // back edges (to a dominating head) and loop bodies; any other retreating edge: irreducible
    let mut body: Vec<Option<Vec<bool>>> = vec![None; n];
    let mut forward_in = vec![0usize; n];
    for &u in &order {
        for v in g.succ[u].targets() {
            if rpo[v] > rpo[u] {
                forward_in[v] += 1;
                continue;
            }
            if !dominates(v, u) {
                return Err(format!("irreducible: block {u} → {v}"));
            }
            let set = body[v].get_or_insert_with(|| {
                let mut s = vec![false; n];
                s[v] = true;
                s
            });
            let mut work = vec![u];
            while let Some(x) = work.pop() {
                if !set[x] {
                    set[x] = true;
                    work.extend(preds[x].iter().copied());
                }
            }
        }
    }
    let size = |h: usize| body[h].as_ref().map_or(0, |s| s.iter().filter(|&&x| x).count());
    let mut heads: Vec<usize> = (0..n).filter(|&h| body[h].is_some()).collect();
    heads.sort_by_key(|&h| size(h));
    // innermost loop containing each block (a head: its own), the loop around each loop
    let mut inner: Vec<Option<usize>> = vec![None; n];
    for &h in &heads {
        let s = body[h].as_ref().unwrap();
        for b in 0..n {
            if s[b] && inner[b].is_none() {
                inner[b] = Some(h);
            }
        }
    }
    let mut outer: Vec<Option<usize>> = vec![None; n];
    for &h in &heads {
        // the smallest other loop containing h
        outer[h] = heads.iter().copied().find(|&o| o != h && body[o].as_ref().unwrap()[h] && size(o) > size(h));
    }
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in order.iter().skip(1) {
        children[idom[b]].push(b);
    }
    // a block outside loops its immediate dominator is in: after the outermost of them
    let mut placed_after = vec![None; n];
    for &w in order.iter().skip(1) {
        let mut l = inner[idom[w]];
        let mut last = None;
        while let Some(h) = l {
            if body[h].as_ref().unwrap()[w] {
                break;
            }
            last = Some(h);
            l = outer[h];
        }
        placed_after[w] = last;
    }
    Ok(Shape { rpo, order, children, body, forward_in, placed_after })
}

/// A label: the head of a loop (a jump to it continues the loop) or the block that follows
/// a `do { } while (false)` (a jump to it leaves the block).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Label {
    Loop(usize),
    Follow(usize),
    /// the end of the program (the mode's epilogue follows)
    Exit,
}

#[derive(Debug)]
pub(super) enum Ast {
    /// a block's instructions (indented for `depth`)
    Code { block: usize, text: String, depth: usize },
    /// the branch at the end of `block`: `then` runs when it is taken (`negated`: when not)
    If { block: usize, cond: String, negated: bool, then: Vec<Ast>, other: Vec<Ast> },
    Loop(usize, Vec<Ast>),
    Block(Label, Vec<Ast>),
    Br(Label),
}

/// Deepest nesting of the emitted constructs (DXC and this translator recurse along it).
const MAX_DEPTH: usize = 160;

struct Builder<'a, 'b> {
    c: &'a Ctx<'a>,
    g: &'a Cfg,
    s: &'a Shape,
    /// None: a dry run (no code; the structure only)
    em: Option<&'b mut emit::Em>,
    labels: Vec<Label>,
    done: Vec<bool>,
    blocks: usize,
}

impl Builder<'_, '_> {
    fn depth(&self) -> usize {
        self.labels.len()
    }

    /// The code of `x` and of the blocks it dominates, appended to `out`.
    fn tree(&mut self, mut x: usize, out: &mut Vec<Ast>) -> Result<(), String> {
        loop {
            match self.tree_step(x, out)? {
                Some(next) => x = next,
                None => return Ok(()),
            }
        }
    }

    /// One step of `tree`: Some(block) when that block's tree follows (a tail, not recursed).
    fn tree_step(&mut self, x: usize, out: &mut Vec<Ast>) -> Result<Option<usize>, String> {
        if self.depth() > MAX_DEPTH {
            return Err("nesting too deep".into());
        }
        let s = self.s;
        // merges dominated by x, placed inside x's construct; latest first (outermost block)
        let mut merges: Vec<usize> = s.children[x].iter().copied().filter(|&y| s.placed_after[y].is_none() && s.forward_in[y] > 1).collect();
        merges.sort_by_key(|&y| std::cmp::Reverse(s.rpo[y]));
        if s.body[x].is_none() {
            return self.within(x, &merges, out);
        }
        // a loop head: what leaves the loop and is placed after it goes into blocks around it
        let mut after: Vec<usize> = s.order.iter().copied().filter(|&w| s.placed_after[w] == Some(x)).collect();
        after.sort_by_key(|&w| std::cmp::Reverse(s.rpo[w]));
        self.around_loop(x, &after, &merges, out)
    }

    fn around_loop(&mut self, x: usize, after: &[usize], merges: &[usize], out: &mut Vec<Ast>) -> Result<Option<usize>, String> {
        let Some((&w, rest)) = after.split_first() else {
            self.labels.push(Label::Loop(x));
            let mut body = Vec::new();
            let mut tail = self.within(x, merges, &mut body)?;
            while let Some(t) = tail {
                tail = self.tree_step(t, &mut body)?;
            }
            self.labels.pop();
            out.push(Ast::Loop(x, body));
            return Ok(None);
        };
        self.labels.push(Label::Follow(w));
        let mut inside = Vec::new();
        let mut tail = self.around_loop(x, rest, merges, &mut inside)?;
        while let Some(t) = tail {
            tail = self.tree_step(t, &mut inside)?;
        }
        self.labels.pop();
        out.push(Ast::Block(Label::Follow(w), inside));
        Ok(Some(w))
    }

    /// x's code with the merges `ys` (latest first) after it, each reached by leaving a block.
    fn within(&mut self, x: usize, ys: &[usize], out: &mut Vec<Ast>) -> Result<Option<usize>, String> {
        let Some((&y, rest)) = ys.split_first() else {
            return self.code(x, out);
        };
        self.labels.push(Label::Follow(y));
        let mut inside = Vec::new();
        let mut tail = self.within(x, rest, &mut inside)?;
        while let Some(t) = tail {
            tail = self.tree_step(t, &mut inside)?;
        }
        self.labels.pop();
        out.push(Ast::Block(Label::Follow(y), inside));
        Ok(Some(y))
    }

    /// Block x and its branch.
    fn code(&mut self, x: usize, out: &mut Vec<Ast>) -> Result<Option<usize>, String> {
        if std::mem::replace(&mut self.done[x], true) {
            return Err(format!("block {x} emitted twice"));
        }
        self.blocks += 1;
        let (start, end) = (self.c.leaders[x], block_end(self.c, x));
        let depth = self.depth();
        let mut flow = emit::Flow::Next;
        let mut text = String::new();
        if let Some(em) = self.em.as_deref_mut() {
            em.indent = depth + 1;
            let block = insts_in(self.c, start, end);
            for (k, d) in block.iter().enumerate() {
                let next = block.get(k + 1).map(|n| n.pc).unwrap_or(end);
                flow = em.inst(d, next);
                if !matches!(flow, emit::Flow::Next) {
                    break;
                }
            }
            text = em.take();
        }
        out.push(Ast::Code { block: x, text, depth: depth + 1 });
        match self.g.succ[x] {
            Succ::Exit => {
                out.push(Ast::Br(Label::Exit));
                Ok(None)
            }
            Succ::Jump(t) => self.branch(x, t, out),
            Succ::Cond(t, f) => {
                // a branch the emitter does not know (counted as unsupported) is never taken, as
                // in the `pc` form
                let cond = match flow {
                    emit::Flow::Cond(cond, _) => cond,
                    _ => "false".into(),
                };
                let jump = |b: &Self, t: Target| t.map_or(Some(Label::Exit), |t| b.label_of(x, t));
                let (jt, jf) = (jump(self, t), jump(self, f));
                if let Some(l) = jt.iter().chain(&jf).find(|l| !self.labels.contains(l)) {
                    return Err(format!("block {x}: jump to {l:?} outside its construct"));
                }
                match (jt, jf) {
                    (Some(lt), Some(lf)) => {
                        out.push(Ast::If { block: x, cond, negated: false, then: vec![Ast::Br(lt)], other: vec![] });
                        out.push(Ast::Br(lf));
                        Ok(None)
                    }
                    (Some(lt), None) => {
                        out.push(Ast::If { block: x, cond, negated: false, then: vec![Ast::Br(lt)], other: vec![] });
                        Ok(f)
                    }
                    (None, Some(lf)) => {
                        out.push(Ast::If { block: x, cond, negated: true, then: vec![Ast::Br(lf)], other: vec![] });
                        Ok(t)
                    }
                    (None, None) => {
                        let (t, f) = (t.unwrap(), f.unwrap());
                        let mut then = Vec::new();
                        self.labels.push(Label::Exit); // depth only: an if is not left by jumps
                        let r = self.tree(t, &mut then).and_then(|_| {
                            let mut other = Vec::new();
                            self.tree(f, &mut other).map(|_| other)
                        });
                        self.labels.pop();
                        out.push(Ast::If { block: x, cond, negated: false, then, other: r? });
                        Ok(None)
                    }
                }
            }
        }
    }

    /// The label a branch x → t jumps to, or None: t's code goes right here.
    fn label_of(&self, x: usize, t: usize) -> Option<Label> {
        let s = self.s;
        if s.rpo[t] <= s.rpo[x] {
            Some(Label::Loop(t))
        } else if s.forward_in[t] > 1 || s.placed_after[t].is_some() {
            Some(Label::Follow(t))
        } else {
            None
        }
    }

    fn branch(&mut self, x: usize, t: usize, out: &mut Vec<Ast>) -> Result<Option<usize>, String> {
        match self.label_of(x, t) {
            Some(l) => {
                if !self.labels.contains(&l) {
                    return Err(format!("block {x}: jump to {l:?} outside its construct"));
                }
                out.push(Ast::Br(l));
                Ok(None)
            }
            None => Ok(Some(t)),
        }
    }
}

fn build(c: &Ctx, g: &Cfg, s: &Shape, em: Option<&mut emit::Em>) -> Result<(Vec<Ast>, usize), String> {
    let mut b = Builder { c, g, s, em, labels: vec![Label::Exit], done: vec![false; g.succ.len()], blocks: 0 };
    let mut top = Vec::new();
    b.tree(0, &mut top)?;
    if let Some(x) = s.order.iter().find(|&&x| !b.done[x]) {
        return Err(format!("block {x} not emitted"));
    }
    // blocks no path reaches: through the emitter all the same (their code is dropped), so
    // that the registers, samplers and unsupported instructions it records are those of the
    // other forms, which emit every block
    if let Some(em) = b.em.as_deref_mut() {
        for x in (0..g.succ.len()).filter(|&x| s.rpo[x] == usize::MAX) {
            let (start, end) = (c.leaders[x], block_end(c, x));
            let block = insts_in(c, start, end);
            for (k, d) in block.iter().enumerate() {
                let next = block.get(k + 1).map(|n| n.pc).unwrap_or(end);
                if !matches!(em.inst(d, next), emit::Flow::Next) {
                    break;
                }
            }
            em.take();
        }
    }
    Ok((vec![Ast::Block(Label::Exit, top)], b.blocks))
}

/// The program as structured code (the body of `main`), or why not (then nothing was emitted:
/// a dry run decides first). `blocks`: blocks emitted.
pub(super) fn structure(c: &Ctx, em: &mut emit::Em, blocks: &mut usize) -> Result<String, String> {
    let g = cfg(c)?;
    let s = shape(&g)?;
    build(c, &g, &s, None)?;
    let (ast, n) = build(c, &g, &s, Some(em))?;
    *blocks += n;
    if CHECK.load(std::sync::atomic::Ordering::Relaxed) || std::env::var_os("XCL_RECOMP_CHECK").is_some() {
        check(&g, &ast)?;
    }
    Ok(render(&ast))
}

#[cfg(test)]
thread_local! {
    /// Tests: this thread translates every program with branches in the general form.
    pub(super) static GENERAL_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn general_only() -> bool {
    #[cfg(test)]
    return GENERAL_ONLY.with(|g| g.get());
    #[cfg(not(test))]
    false
}

/// Tests (or `XCL_RECOMP_CHECK=1`): every structured program is walked against its control
/// flow graph (`check`).
pub(super) static CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// ---------------------------------------------------------------- HLSL

/// Labels jumped to (not by falling off the end) inside `nodes`; `tail`: where falling off the
/// end goes (a jump there as the last statement is left out). Fills `wrapped` with the blocks
/// that need their `do { } while (false)`.
fn refs(nodes: &[Ast], tail: Option<Label>, wrapped: &mut BTreeSet<Label>) -> BTreeSet<Label> {
    let mut r = BTreeSet::new();
    for (k, n) in nodes.iter().enumerate() {
        let t = if k + 1 == nodes.len() { tail } else { None };
        match n {
            Ast::Code { .. } => {}
            Ast::Br(l) => {
                if t != Some(*l) {
                    r.insert(*l);
                }
            }
            Ast::If { then, other, .. } => {
                r.extend(refs(then, t, wrapped));
                r.extend(refs(other, t, wrapped));
            }
            Ast::Loop(h, body) => {
                let mut x = refs(body, None, wrapped);
                x.remove(&Label::Loop(*h));
                r.extend(x);
            }
            Ast::Block(l, body) => {
                let mut x = refs(body, Some(*l), wrapped);
                if x.remove(l) {
                    wrapped.insert(*l);
                }
                r.extend(x);
            }
        }
    }
    r
}

/// Can control fall off the end of `nodes`?
fn falls_off(nodes: &[Ast]) -> bool {
    match nodes.last() {
        None => true,
        Some(Ast::Br(_)) => false,
        Some(Ast::If { then, other, .. }) => falls_off(then) || falls_off(other),
        Some(_) => true,
    }
}

struct Render {
    wrapped: BTreeSet<Label>,
    /// the HLSL loops around the current point: loops and wrapped blocks
    open: Vec<Label>,
    out: String,
    uses_br: bool,
}

fn label_id(l: Label) -> u32 {
    match l {
        Label::Loop(h) => 2 * h as u32 + 2,
        Label::Follow(b) => 2 * b as u32 + 3,
        Label::Exit => 1,
    }
}

impl Render {
    fn pad(&self, depth: usize) -> String {
        "    ".repeat(depth)
    }

    /// Statements of `nodes`; the labels jumped to that are outside them.
    fn nodes(&mut self, nodes: &[Ast], tail: Option<Label>, depth: usize) -> BTreeSet<Label> {
        let mut esc = BTreeSet::new();
        for (k, n) in nodes.iter().enumerate() {
            let t = if k + 1 == nodes.len() { tail } else { None };
            let pad = self.pad(depth);
            match n {
                Ast::Code { text, depth: at, .. } => {
                    // the instructions were indented for the nesting of the builder, where
                    // every block is a level
                    for line in text.lines() {
                        let k = line.len() - line.trim_start_matches(' ').len();
                        let _ = writeln!(self.out, "{pad}{}", &line[k.min(at * 4)..]);
                    }
                }
                Ast::Br(l) => {
                    if t == Some(*l) {
                        continue;
                    }
                    esc.insert(*l);
                    let Some(&top) = self.open.last() else { continue };
                    if top == *l {
                        let s = if matches!(l, Label::Loop(_)) { "continue" } else { "break" };
                        let _ = writeln!(self.out, "{pad}{s};");
                    } else {
                        self.uses_br = true;
                        let _ = writeln!(self.out, "{pad}xcl_br = {}u; break;", label_id(*l));
                    }
                }
                Ast::If { cond, negated, then, other, .. } => {
                    let cond = if *negated { format!("!({cond})") } else { cond.clone() };
                    // a single jump: one line
                    if let ([Ast::Br(l)], []) = (then.as_slice(), other.as_slice()) {
                        if t != Some(*l) && self.open.last() == Some(l) {
                            esc.insert(*l);
                            let s = if matches!(l, Label::Loop(_)) { "continue" } else { "break" };
                            let _ = writeln!(self.out, "{pad}if ({cond}) {s};");
                            continue;
                        }
                    }
                    let _ = writeln!(self.out, "{pad}if ({cond}) {{");
                    esc.extend(self.nodes(then, t, depth + 1));
                    if !other.is_empty() {
                        let _ = writeln!(self.out, "{pad}}} else {{");
                        esc.extend(self.nodes(other, t, depth + 1));
                    }
                    let _ = writeln!(self.out, "{pad}}}");
                }
                Ast::Loop(h, body) => {
                    let l = Label::Loop(*h);
                    let _ = writeln!(self.out, "{pad}[loop] for (;;) {{\n{pad}    {LOOP_GUARD}");
                    self.open.push(l);
                    let mut x = self.nodes(body, None, depth + 1);
                    if falls_off(body) {
                        let _ = writeln!(self.out, "{pad}    break;");
                    }
                    self.open.pop();
                    let _ = writeln!(self.out, "{pad}}}");
                    x.remove(&l);
                    self.pass_on(&x, depth);
                    esc.extend(x);
                }
                Ast::Block(l, body) => {
                    if !self.wrapped.contains(l) {
                        let mut x = self.nodes(body, Some(*l), depth);
                        x.remove(l);
                        esc.extend(x);
                        continue;
                    }
                    let _ = writeln!(self.out, "{pad}do {{");
                    self.open.push(*l);
                    let mut x = self.nodes(body, Some(*l), depth + 1);
                    self.open.pop();
                    let _ = writeln!(self.out, "{pad}}} while (false);");
                    x.remove(l);
                    self.pass_on(&x, depth);
                    esc.extend(x);
                }
            }
        }
        esc
    }

    /// After a loop / wrapped block that jumps `esc` left through `xcl_br`: the enclosing
    /// one takes its own and passes the others on.
    fn pass_on(&mut self, esc: &BTreeSet<Label>, depth: usize) {
        if esc.is_empty() {
            return;
        }
        let pad = self.pad(depth);
        let Some(&p) = self.open.last() else { return };
        let go = if matches!(p, Label::Loop(_)) { "continue" } else { "break" };
        if !esc.contains(&p) {
            let _ = writeln!(self.out, "{pad}if (xcl_br != 0u) break;");
        } else if esc.len() == 1 {
            let _ = writeln!(self.out, "{pad}if (xcl_br != 0u) {{ xcl_br = 0u; {go}; }}");
        } else {
            let _ = writeln!(self.out, "{pad}if (xcl_br != 0u) {{ if (xcl_br == {}u) {{ xcl_br = 0u; {go}; }} break; }}", label_id(p));
        }
    }
}

fn render(ast: &[Ast]) -> String {
    let mut wrapped = BTreeSet::new();
    refs(ast, None, &mut wrapped);
    let mut r = Render { wrapped, open: Vec::new(), out: String::new(), uses_br: false };
    r.nodes(ast, None, 1);
    if r.uses_br {
        format!("    uint xcl_br = 0u;\n{}", r.out)
    } else {
        r.out
    }
}

// ---------------------------------------------------------------- self-check

/// Walks the control flow graph and the structured form with the same branch decisions
/// (pseudo-random, several runs) and compares the blocks they run.
fn check(g: &Cfg, ast: &[Ast]) -> Result<(), String> {
    const STEPS: usize = 20_000;
    for run in 0..24u64 {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ run.wrapping_mul(0xD1B5_4A32_D192_ED03);
        // runs differ in how often branches are taken (loops that run long or short)
        let bias = [128u64, 32, 224, 8, 248, 96][run as usize % 6];
        let mut decide = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 56) < bias
        };
        let mut want = Vec::new();
        let mut decisions = Vec::new();
        let mut b = Some(0usize);
        while let Some(x) = b {
            if want.len() == STEPS {
                break;
            }
            want.push(x);
            b = match g.succ[x] {
                Succ::Exit => None,
                Succ::Jump(t) => Some(t),
                Succ::Cond(t, f) => {
                    let d = decide();
                    decisions.push((x, d));
                    if d {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        let mut walk = Walk { trace: Vec::new(), decisions: &decisions, next: 0 };
        match walk.nodes(ast) {
            Err(e) if e != "steps" => return Err(format!("structured form, run {run}: {e}")),
            Ok(Out::Br(l)) if l != Label::Exit => return Err(format!("structured form, run {run}: leaves through {l:?}")),
            _ => {}
        }
        let n = want.len().min(walk.trace.len());
        if walk.trace[..n] != want[..n] || (want.len() < STEPS && walk.trace.len() != want.len()) {
            let k = (0..n).find(|&k| walk.trace[k] != want[k]).unwrap_or(n);
            return Err(format!(
                "structured form, run {run}: step {k} runs block {:?}, the program block {:?}",
                walk.trace.get(k),
                want.get(k)
            ));
        }
    }
    Ok(())
}

enum Out {
    Fall,
    Br(Label),
}

struct Walk<'a> {
    trace: Vec<usize>,
    /// (block, taken) of each conditional branch the graph walk took
    decisions: &'a [(usize, bool)],
    next: usize,
}

impl Walk<'_> {
    fn nodes(&mut self, nodes: &[Ast]) -> Result<Out, String> {
        for n in nodes {
            match n {
                Ast::Code { block, .. } => {
                    if self.trace.len() == 20_000 {
                        return Err("steps".into());
                    }
                    self.trace.push(*block);
                }
                Ast::If { block, negated, then, other, .. } => {
                    let Some(&(b, d)) = self.decisions.get(self.next) else { return Err("steps".into()) };
                    if b != *block {
                        return Err(format!("the branch of block {block} takes the decision of block {b}"));
                    }
                    self.next += 1;
                    if let Out::Br(l) = self.nodes(if d != *negated { then } else { other })? {
                        return Ok(Out::Br(l));
                    }
                }
                Ast::Br(l) => return Ok(Out::Br(*l)),
                Ast::Loop(h, body) => loop {
                    match self.nodes(body)? {
                        Out::Br(Label::Loop(x)) if x == *h => continue,
                        Out::Br(l) => return Ok(Out::Br(l)),
                        Out::Fall => return Err(format!("loop {h} falls off its end")),
                    }
                },
                Ast::Block(l, body) => match self.nodes(body)? {
                    Out::Br(x) if x != *l => return Ok(Out::Br(x)),
                    _ => {}
                },
            }
        }
        Ok(Out::Fall)
    }
}
