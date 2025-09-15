use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: OperatingSystem,
    pub architecture: Architecture,
    pub has_cuda: bool,
    pub has_opencl: bool,
    pub has_vulkan: bool,
    pub kernel_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperatingSystem {
    Linux,
    Windows,
    MacOS,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    ARM64,
    X86,
    Unknown,
}

pub fn detect_platform() -> Result<PlatformInfo> {
    let os = detect_operating_system();
    let architecture = detect_architecture();
    let has_cuda = check_cuda_availability();
    let has_opencl = check_opencl_availability();
    let has_vulkan = check_vulkan_availability();
    let kernel_version = get_kernel_version();

    Ok(PlatformInfo {
        os,
        architecture,
        has_cuda,
        has_opencl,
        has_vulkan,
        kernel_version,
    })
}

fn detect_operating_system() -> OperatingSystem {
    #[cfg(target_os = "linux")]
    return OperatingSystem::Linux;
    
    #[cfg(target_os = "windows")]
    return OperatingSystem::Windows;
    
    #[cfg(target_os = "macos")]
    return OperatingSystem::MacOS;
    
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    return OperatingSystem::Unknown;
}

fn detect_architecture() -> Architecture {
    #[cfg(target_arch = "x86_64")]
    return Architecture::X86_64;
    
    #[cfg(target_arch = "aarch64")]
    return Architecture::ARM64;
    
    #[cfg(target_arch = "x86")]
    return Architecture::X86;
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "x86")))]
    return Architecture::Unknown;
}

fn check_cuda_availability() -> bool {
    // Check for CUDA driver/runtime
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    #[cfg(target_os = "windows")]
    {
        // Check for CUDA DLLs or nvidia-smi.exe
        std::process::Command::new("nvidia-smi.exe")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    false
}

fn check_opencl_availability() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check for OpenCL ICD loader
        std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so").exists() ||
        std::path::Path::new("/usr/lib/libOpenCL.so").exists()
    }
    
    #[cfg(target_os = "windows")]
    {
        // Check Windows OpenCL installation
        std::path::Path::new("C:/Windows/System32/OpenCL.dll").exists()
    }
    
    #[cfg(target_os = "macos")]
    {
        // OpenCL is available as part of the OpenCL framework on macOS
        std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    false
}

fn check_vulkan_availability() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check for Vulkan loader
        std::process::Command::new("vulkaninfo")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    #[cfg(target_os = "windows")]
    {
        // Check for Vulkan DLL
        std::path::Path::new("C:/Windows/System32/vulkan-1.dll").exists()
    }
    
    #[cfg(target_os = "macos")]
    {
        // Check for MoltenVK or Vulkan SDK
        std::path::Path::new("/usr/local/lib/libvulkan.dylib").exists()
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    false
}

fn get_kernel_version() -> String {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("uname")
            .arg("-r")
            .output()
            .and_then(|output| {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    Ok("unknown".to_string())
                }
            })
            .unwrap_or_else(|_| "unknown".to_string())
    }
    
    #[cfg(target_os = "windows")]
    {
        // Get Windows version
        "Windows".to_string() // Would use Windows API for actual version
    }
    
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("uname")
            .arg("-r")
            .output()
            .and_then(|output| {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    Ok("unknown".to_string())
                }
            })
            .unwrap_or_else(|_| "unknown".to_string())
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform_info = detect_platform().unwrap();
        
        // Basic validation
        match platform_info.os {
            OperatingSystem::Unknown => panic!("Should detect operating system"),
            _ => {}
        }
        
        match platform_info.architecture {
            Architecture::Unknown => panic!("Should detect architecture"),
            _ => {}
        }
        
        assert!(!platform_info.kernel_version.is_empty());
    }

    #[test] 
    fn test_operating_system_detection() {
        let os = detect_operating_system();
        
        #[cfg(target_os = "linux")]
        assert!(matches!(os, OperatingSystem::Linux));
        
        #[cfg(target_os = "windows")]
        assert!(matches!(os, OperatingSystem::Windows));
        
        #[cfg(target_os = "macos")]
        assert!(matches!(os, OperatingSystem::MacOS));
    }

    #[test]
    fn test_architecture_detection() {
        let arch = detect_architecture();
        
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(arch, Architecture::X86_64));
        
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(arch, Architecture::ARM64));
        
        #[cfg(target_arch = "x86")]
        assert!(matches!(arch, Architecture::X86));
    }
}