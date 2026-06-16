use indicatif::{ProgressBar, ProgressStyle};

pub fn denoising_bar(n_steps: usize) -> ProgressBar {
    let pb = ProgressBar::new(n_steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} Denoising [{bar:20.green/white.dim}] {pos}/{len}  {msg}  elapsed {elapsed_precise}",
        )
        .expect("valid indicatif template")
        .progress_chars("█░")
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}
