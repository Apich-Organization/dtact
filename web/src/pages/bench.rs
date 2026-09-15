use leptos::prelude::*;

/// Live Criterion HTML report, deployed by `.github/workflows/deploy.yml`
/// on every push to `main` (job `build-bench` runs `cargo bench --
/// --save-baseline main`, `deploy` job publishes `target/criterion` under
/// this path). Always reflects the latest commit on `main`.
const BENCH_URL: &str = "https://dtact.apich.org/bench/report/";

/// The workflow that produces `BENCH_URL`, rather than a link to one
/// specific (and eventually stale) run of it.
const BENCH_WORKFLOW_URL: &str =
    "https://github.com/Apich-Organization/dtact/actions/workflows/deploy.yml";

/// PR-time regression check — compares a branch's run against the `main`
/// baseline `deploy.yml` saved, per `.github/workflows/benchmark.yml`.
const BENCH_PR_WORKFLOW_URL: &str =
    "https://github.com/Apich-Organization/dtact/actions/workflows/benchmark.yml";

const BENCH_SOURCE_URL: &str =
    "https://github.com/Apich-Organization/dtact/tree/main/benches/scheduler_efficiency.rs";

#[component]
pub fn BenchPage() -> impl IntoView {
    view! {
        <div class="bench-page pt-nav page-wrap">

            // ── Header ───────────────────────────────────────────────────
            <section class="section">
                <span class="section-chip">"Performance Measurements"</span>
                <h1 class="bench-title">"Benchmarks"</h1>
                <p class="mt-sm" style="max-width:64ch">
                    "Criterion-based engineering benchmarks (`benches/scheduler_efficiency.rs`)
                     comparing dtact against Tokio: spawn+join throughput, cooperative-yield
                     fast path, hot-core work deflection, cancellation, and panic-vs-normal
                     completion overhead. The figures below are a fixed reference snapshot;
                     the live report always reflects the latest commit on "
                    <code class="mono">"main"</code>"."
                </p>
                <div class="flex gap-sm mt-md flex-wrap">
                    <a href=BENCH_URL target="_blank" rel="noopener" class="btn btn-primary">
                        "\u{1F4CA} Open Live Report \u{2197}"
                    </a>
                    <a href=BENCH_WORKFLOW_URL target="_blank" rel="noopener" class="btn btn-ghost">
                        "CI/CD Workflow \u{2197}"
                    </a>
                    <a href=BENCH_SOURCE_URL target="_blank" rel="noopener" class="btn btn-ghost">
                        "Bench Source"
                    </a>
                </div>
            </section>

            // ── Snapshot banner ──────────────────────────────────────────
            <section class="section">
                <div class="bench-snapshot glass card-pad">
                    <div class="bench-snapshot-row">
                        <span class="section-chip">"Data Snapshot"</span>
                        <span class="bench-snapshot-meta">
                            "4-core x86_64 dev machine (not the CI runner)\u{00A0}
                             \u{00B7}\u{00A0}4 workers each\u{00A0}
                             \u{00B7}\u{00A0}50 samples / 5\u{00A0}s measurement / 2\u{00A0}s warm-up"
                        </span>
                    </div>
                    <p class="text-sm text-muted mt-sm">
                        "Absolute numbers here reflect a quiet local machine and will differ from "
                        <a href=BENCH_URL target="_blank" rel="noopener" class="mono">
                            "dtact.apich.org/bench"
                        </a>
                        ", which runs on a shared GitHub-hosted runner. Every PR is additionally
                         checked against the "
                        <a href=BENCH_PR_WORKFLOW_URL target="_blank" rel="noopener" class="mono">
                            "main baseline"
                        </a>
                        " for regressions before merge."
                    </p>
                </div>
            </section>

            // ── Spawn + Join ──────────────────────────────────────────────
            <section class="section">
                <BenchFigure
                    title="Spawn + Join Throughput"
                    chip="bench_spawn_join \u{2014} spawn N tasks, run, join all"
                    src="spawn_join.png"
                    alt="Spawn+join per-task cost and dtact/Tokio advantage ratio across N=1,000..100,000"
                    caption="Left: median time per task (total time \u{00F7} N), log-x. Right: Tokio's
                             per-task cost divided by dtact's \u{2014} above 1.0 means dtact is faster."
                />
            </section>

            // ── Work Deflection ──────────────────────────────────────────
            <section class="section">
                <BenchFigure
                    title="Work Deflection (Hot Core)"
                    chip="bench_deflection \u{2014} all tasks spawned from one core"
                    src="work_deflection.png"
                    alt="Work deflection per-task cost and dtact/Tokio advantage ratio across N=1,000..100,000"
                    caption="Every task is spawned onto a single worker; dtact's P2P mesh
                             redistributes the backlog to idle peers. Same axes as above."
                />
            </section>

            // ── Where the advantage comes from ──────────────────────────
            <section class="section">
                <div class="glass card-pad-lg">
                    <span class="section-chip">"Reading the Curve"</span>
                    <h2 class="mt-sm">"Where the Advantage Comes From"</h2>
                    <p class="mt-sm text-sm">
                        "Both figures show the same non-monotonic shape: dtact's advantage is
                         largest at the extremes of "<code class="mono">"N"</code>" and narrows \u{2014}
                         in Work Deflection, briefly dipping just below 1\u{00D7} \u{2014} in the
                         N\u{2248}4,000\u{2013}16,000 range. Two effects are stacked on top of each
                         other:"
                    </p>
                    <ul class="community-list mt-sm">
                        <li>
                            "At low N, Tokio pays a largely fixed per-batch setup cost (runtime
                             + task-set bookkeeping) that dominates the median; this amortizes
                             away by roughly N=4,000, closing most of dtact's early lead."
                        </li>
                        <li>
                            "At high N, Tokio's per-task cost starts growing again. We measured
                             this directly with "<code class="mono">"perf stat -e page-faults"</code>
                             " across the same N sweep: dtact's fault count stayed essentially flat
                             (536\u{2192}831 over the full range), while Tokio's grew by two orders
                             of magnitude (867\u{2192}143,547) \u{2014} consistent with allocator
                             pressure from its per-task heap allocations, against dtact's
                             fixed-capacity slot arena. We don't have hardware cache-miss counters
                             in this environment (a Xen-virtualized VM exposes only software
                             perf events), so this is the best-supported explanation we have
                             rather than a fully instrumented one; an earlier L2/L3
                             cache-hierarchy hypothesis was tested and did not hold up against
                             this data."
                        </li>
                    </ul>
                </div>
            </section>

            // ── Yield Fast Path ──────────────────────────────────────────
            <section class="section">
                <div class="grid-2">
                    <BenchFigure
                        title="Yield Fast Path"
                        chip="bench_yield_now_loop \u{2014} 10 tasks \u{00D7} 100 yield_now() each"
                        src="yield_fast_path.png"
                        alt="Yield fast path: dtact 93 microseconds vs Tokio 572 microseconds"
                        caption="Self-yield with nothing else ready to interleave with."
                    />
                    <div class="glass card-pad bench-summary-card">
                        <span class="section-chip">"Why dtact Wins Here"</span>
                        <h3 class="mt-sm">"Fast Path vs. Real Switch"</h3>
                        <p class="text-sm text-muted mt-sm">
                            "`wait_pinned` re-polls a still-pending future up to "
                            <code class="mono">"adaptive_spin_count"</code>" times before ever
                             performing a real assembly context switch. A self "
                            <code class="mono">"yield_now()"</code>" resolves on its second poll,
                             so when nothing else needs this core, it never reaches the real
                             suspend/resume path \u{2014} one atomic swap and a couple of extra
                             polls. That's what this benchmark measures."
                        </p>
                        <p class="text-sm text-muted mt-sm">
                            "That's a different cost from "
                            <code class="mono">"yield_to()"</code>"'s explicit handoff to a named
                             peer fiber (`bench_fiber_pingpong`), which does take the real wake
                             protocol and, when the peer isn't already running, a real switch \u{2014}
                             the case where dtact's stackful fibers (full call-stack save/restore)
                             are genuinely heavier than a stackless poll. That benchmark is
                             tracked in the live report but has no Tokio-equivalent measurement
                             here, since Tokio has no directly comparable named-peer handoff
                             primitive."
                        </p>
                    </div>
                </div>
            </section>

            // ── Cancellation & panic overhead ────────────────────────────
            <section class="section">
                <div class="glass card-pad-lg">
                    <span class="section-chip">"Also Tracked"</span>
                    <h2 class="mt-sm">"Cancellation & Panic Overhead"</h2>
                    <p class="mt-sm text-sm">
                        "`bench_cancellation` (cost of `cancel()` racing a fiber's own
                         completion) and `bench_panic_vs_normal_completion` (a panicking task's
                         `catch_unwind` overhead against a normal return) run on every push and
                         every PR alongside the benchmarks above. We don't ship fixed figures
                         for them here since they're closer to overhead-regression checks than
                         to a dtact/Tokio comparison \u{2014} see the "
                        <a href=BENCH_URL target="_blank" rel="noopener" class="mono">"live report"</a>
                        " for current numbers."
                    </p>
                </div>
            </section>

            // ── Methodology ──────────────────────────────────────────────
            <section class="section">
                <div class="glass card-pad-lg">
                    <span class="section-chip">"Methodology"</span>
                    <h2>"Benchmark Configuration"</h2>
                    <div class="grid-2 mt-md">
                        <div>
                            <p class="algo-section-title">"Criterion Settings"</p>
                            <pre class="code-block text-xs">
"criterion_group!(
    name   = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(50)
        .noise_threshold(0.05);
    targets =
        bench_spawn_join,
        bench_yield_now_loop,
        bench_fiber_pingpong,
        bench_cancellation,
        bench_panic_vs_normal_completion,
        bench_deflection,
);"
                            </pre>
                        </div>
                        <div>
                            <p class="algo-section-title">"CI Environment"</p>
                            <table class="cfg-table">
                                <thead><tr><th>"Property"</th><th>"Value"</th></tr></thead>
                                <tbody>
                                    <tr><td>"Runner"</td><td>"ubuntu-latest (GitHub hosted)"</td></tr>
                                    <tr><td>"Workers"</td><td>"4 (dtact and Tokio)"</td></tr>
                                    <tr><td>"Sampling"</td><td>"50 samples, 5 s measurement, 2 s warm-up"</td></tr>
                                    <tr><td>"Noise threshold"</td><td>"5 % (results below filtered)"</td></tr>
                                    <tr><td>"Baseline"</td><td>"named \u{201C}main\u{201D}, saved on every push"</td></tr>
                                    <tr><td>"PR check"</td><td>"compared against main via --baseline-lenient"</td></tr>
                                    <tr><td>"Report"</td><td>"Criterion HTML + JSON"</td></tr>
                                </tbody>
                            </table>
                            <p class="text-sm text-muted mt-sm">
                                "GitHub-hosted runners share physical hardware \u{2014} trends and
                                 dtact/Tokio ratios are the reliable signal; absolute numbers
                                 reflect CI conditions and the local-machine figures above will
                                 not match them exactly."
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            // ── CTA ───────────────────────────────────────────────────────
            <section class="section">
                <div class="bench-cta glass card-pad">
                    <h3>"\u{1F4CA} Full Criterion Report"</h3>
                    <p class="mt-sm">
                        "The live report includes per-benchmark violin plots, regression
                         detection, and historical trend data. Always reflects the latest commit."
                    </p>
                    <a href=BENCH_URL target="_blank" rel="noopener"
                       class="btn btn-primary mt-md">
                        "Open Benchmark Report \u{2197}"
                    </a>
                </div>
            </section>
        </div>
    }
}

// ── Chart / figure container ─────────────────────────────────────────────────

#[component]
fn BenchFigure(
    title: &'static str,
    chip: &'static str,
    src: &'static str,
    alt: &'static str,
    caption: &'static str,
) -> impl IntoView {
    view! {
        <div class="glass card-pad bench-chart-card">
            <div class="bench-chart-head">
                <div>
                    <span class="section-chip">{chip}</span>
                    <h3 class="mt-sm">{title}</h3>
                </div>
                <div class="bench-legend">
                    <span class="bench-legend-item bench-legend-dtact">"dtact"</span>
                    <span class="bench-legend-item bench-legend-tokio">"Tokio"</span>
                </div>
            </div>
            <figure class="bench-figure mt-lg">
                <img src=src alt=alt />
                <figcaption class="text-xs text-muted mt-sm">{caption}</figcaption>
            </figure>
        </div>
    }
}
