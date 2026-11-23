//! The sample marker must survive libtest capture without becoming a GPU requirement.

#[test]
#[ignore]
fn sample_skip_fixture() {
    xcl::gpu::recomp::runner::skip_sample("XCL_SAMPLE_TEST_FIXTURE");
}

#[test]
fn sample_skips_are_visible_without_nocapture_and_independent_of_gpu_requirements() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "sample_skip_fixture"])
        .env("XCL_REQUIRE_GPU", "wave64,mesh")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("1 passed"), "{stdout}");
    assert_eq!(stderr.trim(), "SKIPPED_SAMPLE sample_skip_fixture: XCL_SAMPLE_TEST_FIXTURE not set");
}
