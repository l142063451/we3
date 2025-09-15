use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};

mod installer;
mod platform;
mod config;

use installer::*;
use platform::*;
use config::*;

#[derive(Parser)]
#[command(name = "vgpu-install")]
#[command(about = "WE3 Virtual GPU Installation and Configuration Tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    #[arg(long, default_value = "/opt/we3-vgpu")]
    pub install_dir: PathBuf,
    
    #[arg(long)]
    pub config_file: Option<PathBuf>,
    
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Install WE3 vGPU system
    Install {
        #[arg(long, default_value = "full")]
        profile: String,
        
        #[arg(long)]
        enable_cuda: bool,
        
        #[arg(long)]
        enable_opencl: bool,
        
        #[arg(long)]
        enable_vulkan: bool,
        
        #[arg(long)]
        skip_verification: bool,
    },
    
    /// Uninstall WE3 vGPU system
    Uninstall {
        #[arg(long)]
        keep_config: bool,
    },
    
    /// Update existing installation
    Update {
        #[arg(long)]
        force: bool,
    },
    
    /// Verify installation
    Verify,
    
    /// Configure vGPU settings
    Configure {
        #[arg(long)]
        compute_units: Option<u32>,
        
        #[arg(long)]
        memory_gb: Option<f64>,
        
        #[arg(long)]
        enable_formal_verification: Option<bool>,
    },
    
    /// Run installation tests
    Test,
    
    /// Show installation status
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_level(true)
            .with_target(true)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_level(false)
            .with_target(false)
            .init();
    }

    info!("WE3 vGPU Installer v1.2.0 starting");

    let platform_info = detect_platform()?;
    info!("Detected platform: {:?}", platform_info);

    let mut installer = VGPUInstaller::new(cli.install_dir.clone(), platform_info)?;

    match cli.command {
        Commands::Install { 
            profile, 
            enable_cuda, 
            enable_opencl, 
            enable_vulkan, 
            skip_verification 
        } => {
            let config = InstallationConfig {
                profile,
                enable_cuda,
                enable_opencl,
                enable_vulkan,
                skip_verification,
                install_directory: cli.install_dir,
                ..Default::default()
            };
            
            installer.install(config).await?;
            info!("Installation completed successfully");
        },
        
        Commands::Uninstall { keep_config } => {
            installer.uninstall(keep_config).await?;
            info!("Uninstallation completed");
        },
        
        Commands::Update { force } => {
            installer.update(force).await?;
            info!("Update completed");
        },
        
        Commands::Verify => {
            let result = installer.verify_installation().await?;
            if result.success {
                info!("Installation verification successful");
            } else {
                error!("Installation verification failed: {:?}", result.issues);
            }
        },
        
        Commands::Configure { 
            compute_units, 
            memory_gb, 
            enable_formal_verification 
        } => {
            let mut config = installer.load_config().await?;
            
            if let Some(cu) = compute_units {
                config.vgpu_config.compute_units = cu;
            }
            if let Some(mem) = memory_gb {
                config.vgpu_config.memory_size_gb = mem;
            }
            if let Some(verify) = enable_formal_verification {
                config.vgpu_config.enable_formal_verification = verify;
            }
            
            installer.save_config(&config).await?;
            info!("Configuration updated");
        },
        
        Commands::Test => {
            let test_results = installer.run_tests().await?;
            info!("Test results: {:?}", test_results);
        },
        
        Commands::Status => {
            let status = installer.get_status().await?;
            println!("{:#?}", status);
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_installer_creation() {
        let temp_dir = tempdir().unwrap();
        let platform_info = PlatformInfo {
            os: OperatingSystem::Linux,
            architecture: Architecture::X86_64,
            has_cuda: false,
            has_opencl: false,
            has_vulkan: false,
            kernel_version: "5.4.0".to_string(),
        };
        
        let installer = VGPUInstaller::new(temp_dir.path().to_path_buf(), platform_info);
        assert!(installer.is_ok());
    }
}