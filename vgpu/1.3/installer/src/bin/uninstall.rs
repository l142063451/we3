//! vGPU v1.3 Infinite FLOPS Framework - Uninstaller Binary
//!
//! Clean removal of vGPU components with optional system cleanup.

use anyhow::Result;
use clap::{Arg, Command as ClapCommand};
use log::{info, error};
use std::path::PathBuf;
use vgpu_installer::VGPUUninstaller;

fn main() -> Result<()> {
    env_logger::init();

    let matches = ClapCommand::new("vgpu-uninstall")
        .version("1.3.0")
        .about("vGPU v1.3 Infinite FLOPS Framework - Uninstaller")
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .value_name("PATH")
                .help("Installation path to remove")
                .default_value("/usr/local/vgpu")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("clean-all")
                .long("clean-all")
                .help("Remove all vGPU configuration and data")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("force")
                .short('f')
                .long("force")
                .help("Force removal without confirmation")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    println!("🗑️  vGPU v1.3 Infinite FLOPS Framework - Uninstaller");
    println!("==================================================");

    let installation_path = matches.get_one::<PathBuf>("path").unwrap();
    let mut uninstaller = VGPUUninstaller::new(installation_path.clone());

    if matches.get_flag("clean-all") {
        uninstaller.set_clean_all(true);
    }

    // Display what will be removed
    display_removal_plan(&uninstaller, installation_path)?;

    // Confirm removal
    if !matches.get_flag("force") {
        if !confirm_uninstallation()? {
            println!("❌ Uninstallation cancelled by user");
            return Ok(());
        }
    }

    // Perform uninstallation
    match uninstaller.uninstall() {
        Ok(()) => {
            println!("✅ vGPU v1.3 uninstalled successfully!");
        },
        Err(e) => {
            error!("❌ Uninstallation failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn display_removal_plan(_uninstaller: &VGPUUninstaller, path: &PathBuf) -> Result<()> {
    println!();
    println!("📋 Removal Plan");
    println!("---------------");
    println!("Installation Path: {}", path.display());
    
    if path.exists() {
        let metadata = std::fs::metadata(path)?;
        println!("Directory exists: {} bytes", metadata.len());
    } else {
        println!("Directory does not exist");
    }

    println!();
    println!("Will remove:");
    println!("  - vGPU libraries and binaries");
    println!("  - API interception configuration");
    println!("  - Mathematical engine data");
    println!("  - System services (if any)");

    Ok(())
}

fn confirm_uninstallation() -> Result<bool> {
    use std::io::{self, Write};
    
    print!("\n❗ This will permanently remove vGPU v1.3. Continue? [y/N]: ");
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let response = input.trim().to_lowercase();
    Ok(response == "y" || response == "yes")
}