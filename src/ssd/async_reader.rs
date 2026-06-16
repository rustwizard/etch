use crate::ssd::offset_index::TensorMeta;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

pub enum IoRequest {
    LoadLayer {
        layer_id: usize,
        tensor_metas: Vec<(String, TensorMeta)>,
    },
    Shutdown,
}

pub enum IoResult {
    LayerReady {
        layer_id: usize,
        buffers: Vec<(String, Vec<u8>)>,
    },
    Error {
        layer_id: usize,
        message: String,
    },
}

pub struct AsyncReader {
    tx: SyncSender<IoRequest>,
    rx: Receiver<IoResult>,
}

impl AsyncReader {
    pub fn spawn(path: &std::path::Path) -> Self {
        let (io_tx, io_rx) = sync_channel::<IoRequest>(2);
        let (res_tx, res_rx) = sync_channel::<IoResult>(2);
        let file_path = path.to_path_buf();

        std::thread::spawn(move || {
            for req in io_rx {
                match req {
                    IoRequest::LoadLayer {
                        layer_id,
                        tensor_metas,
                    } => {
                        let result = read_layer(&file_path, &tensor_metas);
                        match result {
                            Ok(buffers) => {
                                let _ = res_tx.send(IoResult::LayerReady { layer_id, buffers });
                            }
                            Err(e) => {
                                let _ = res_tx.send(IoResult::Error {
                                    layer_id,
                                    message: e.to_string(),
                                });
                            }
                        }
                    }
                    IoRequest::Shutdown => break,
                }
            }
        });

        Self {
            tx: io_tx,
            rx: res_rx,
        }
    }

    pub fn request_layer(&self, layer_id: usize, tensors: Vec<(String, TensorMeta)>) {
        self.tx
            .send(IoRequest::LoadLayer {
                layer_id,
                tensor_metas: tensors,
            })
            .expect("async reader channel closed");
    }

    pub fn wait_layer(&self) -> IoResult {
        self.rx.recv().expect("async reader channel closed")
    }

    pub fn shutdown(self) {
        let _ = self.tx.send(IoRequest::Shutdown);
    }
}

fn read_layer(
    path: &std::path::Path,
    tensors: &[(String, TensorMeta)],
) -> std::io::Result<Vec<(String, Vec<u8>)>> {
    let file = File::open(path)?;
    let mut buffers = Vec::with_capacity(tensors.len());

    for (name, meta) in tensors {
        let mut buf = vec![0u8; meta.size_bytes as usize];
        file.read_exact_at(&mut buf, meta.offset_bytes)?;
        buffers.push((name.clone(), buf));
    }

    Ok(buffers)
}
