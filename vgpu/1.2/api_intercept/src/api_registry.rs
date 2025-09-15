use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// API Registry for tracking loaded libraries and function pointers
pub struct APIRegistry {
    loaded_libraries: parking_lot::RwLock<HashMap<String, LibraryHandle>>,
    function_registry: dashmap::DashMap<String, FunctionPointer>,
    call_count: AtomicU64,
}

pub struct LibraryHandle {
    pub name: String,
    pub handle: Option<libloading::Library>,
    pub loaded_at: std::time::Instant,
    pub function_count: usize,
}

pub struct FunctionPointer {
    pub name: String,
    pub library: String,
    pub address: Option<usize>,
    pub call_count: AtomicU64,
}

impl APIRegistry {
    pub fn new() -> Self {
        Self {
            loaded_libraries: parking_lot::RwLock::new(HashMap::new()),
            function_registry: dashmap::DashMap::new(),
            call_count: AtomicU64::new(0),
        }
    }

    pub fn register_library(&self, name: &str) -> anyhow::Result<()> {
        let mut libraries = self.loaded_libraries.write();
        
        if !libraries.contains_key(name) {
            // Attempt to load the library (optional for intercepting)
            let handle = match unsafe { libloading::Library::new(name) } {
                Ok(lib) => Some(lib),
                Err(_) => {
                    tracing::debug!("Could not load library {}, continuing with interception only", name);
                    None
                }
            };

            libraries.insert(name.to_string(), LibraryHandle {
                name: name.to_string(),
                handle,
                loaded_at: std::time::Instant::now(),
                function_count: 0,
            });
        }
        
        Ok(())
    }

    pub fn register_function(&self, function_name: &str, library_name: &str) {
        let function_ptr = FunctionPointer {
            name: function_name.to_string(),
            library: library_name.to_string(),
            address: None, // Could be resolved if needed
            call_count: AtomicU64::new(0),
        };

        self.function_registry.insert(function_name.to_string(), function_ptr);
    }

    pub fn increment_call_count(&self, function_name: &str) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        
        if let Some(mut function) = self.function_registry.get_mut(function_name) {
            function.call_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_total_calls(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    pub fn get_function_calls(&self, function_name: &str) -> u64 {
        self.function_registry
            .get(function_name)
            .map(|f| f.call_count.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}