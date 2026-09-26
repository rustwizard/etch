use std::sync::atomic::{AtomicUsize, Ordering};

static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);

/// Install the Ctrl-C handler.
///
/// First press requests a graceful stop: the denoising loops poll
/// [`interrupted`] at step boundaries and bail out, the seed loop in `main`
/// breaks, and the process exits with code 130. Second press exits
/// immediately (useful while a long step, model load, or VAE decode is
/// running and the graceful path is too slow).
pub fn install() {
    let res = ctrlc::set_handler(|| {
        let n = INTERRUPTS.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= 2 {
            std::process::exit(130);
        }
        eprintln!(
            "\nInterrupt received — stopping after the current step. Press Ctrl-C again to force quit."
        );
    });
    if let Err(e) = res {
        tracing::warn!("Failed to install Ctrl-C handler: {e}");
    }
}

/// True once the user has pressed Ctrl-C at least once.
pub fn interrupted() -> bool {
    INTERRUPTS.load(Ordering::SeqCst) > 0
}
