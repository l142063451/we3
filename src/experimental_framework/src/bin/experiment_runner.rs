/*!
# Experiment Runner Binary

Main binary for executing comprehensive experiments on WE3 mathematical frameworks.
*/

use experimental_framework::*;
use experimental_framework::experiments::{ExperimentBuilder, Framework, ParameterValue};
use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "experiment_runner")]
#[command(about = "WE3 Experimental Framework - Large-Scale Experiments & Reproducible Benchmarks")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Output directory for results
    #[arg(short, long, default_value = "./experimental_data")]
    output: PathBuf,
    
    /// Configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single experiment
    Run {
        /// Experiment name
        name: String,
        
        /// Target framework
        #[arg(short, long)]
        framework: Option<String>,
        
        /// Number of repetitions
        #[arg(short, long, default_value = "10")]
        repetitions: usize,
    },
    
    /// Run comprehensive benchmark suite
    Benchmark {
        /// Include all frameworks
        #[arg(long)]
        all_frameworks: bool,
        
        /// Include scaling analysis
        #[arg(long)]
        scaling: bool,
        
        /// Include memory profiling
        #[arg(long)]
        memory: bool,
    },
    
    /// Analyze experimental results
    Analyze {
        /// Pattern for result files
        pattern: String,
    },
    
    /// Generate example experiment configuration
    Example,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize framework
    let config = if let Some(config_path) = cli.config {
        // Load configuration from file
        FrameworkConfig::default()
    } else {
        FrameworkConfig::default()
    };
    
    initialize_framework(Some(config)).await?;
    
    match cli.command {
        Commands::Run { name, framework, repetitions } => {
            run_experiment(name, framework, repetitions, cli.output).await?;
        },
        
        Commands::Benchmark { all_frameworks, scaling, memory } => {
            run_benchmark_suite(all_frameworks, scaling, memory, cli.output).await?;
        },
        
        Commands::Analyze { pattern } => {
            analyze_results(pattern).await?;
        },
        
        Commands::Example => {
            generate_example_config(cli.output).await?;
        },
    }
    
    Ok(())
}

async fn run_experiment(name: String, framework: Option<String>, repetitions: usize, output: PathBuf) -> Result<()> {
    println!("🚀 Running experiment: {}", name);
    
    let target_framework = match framework.as_deref() {
        Some("gf") => Framework::GeneratingFunctions,
        Some("kc") => Framework::KnowledgeCompilation,
        Some("tn") => Framework::TensorNetworks,
        Some("idv") => Framework::IdvBits,
        Some("gi") => Framework::GodIndex,
        Some("hv") => Framework::HybridVerifier,
        Some("vgpu") => Framework::VgpuShim,
        _ => Framework::All,
    };
    
    let experiment = ExperimentBuilder::new(&name)
        .description("Custom experiment run")
        .framework(target_framework)
        .parameter("size", vec![
            ParameterValue::Integer(100),
            ParameterValue::Integer(1000),
            ParameterValue::Integer(10000)
        ])
        .parameter("precision", vec![
            ParameterValue::Float(1e-6),
            ParameterValue::Float(1e-9)
        ])
        .repetitions(repetitions)
        .output_directory(output.clone())
        .build();
    
    let results = experiment.execute().await?;
    
    println!("✅ Experiment completed with {} results", results.len());
    
    // Save results
    let results_file = output.join(format!("{}_results.json", name));
    let json_data = serde_json::to_string_pretty(&results)?;
    std::fs::write(results_file, json_data)?;
    
    Ok(())
}

async fn run_benchmark_suite(all_frameworks: bool, scaling: bool, memory: bool, output: PathBuf) -> Result<()> {
    println!("🏁 Starting benchmark suite");
    
    let mut suite = BenchmarkSuite::new();
    
    if all_frameworks {
        suite = suite.add_framework_benchmarks();
    }
    
    if scaling {
        suite = suite.add_scaling_experiments();
    }
    
    if memory {
        suite = suite.add_memory_profiling();
    }
    
    // If no specific options, run framework benchmarks by default
    if !all_frameworks && !scaling && !memory {
        suite = suite.add_framework_benchmarks();
    }
    
    let results = suite.run_all().await?;
    
    println!("✅ Benchmark suite completed with {} benchmark results", results.len());
    
    // Save benchmark results
    let benchmark_file = output.join("benchmark_results.json");
    let json_data = serde_json::to_string_pretty(&results)?;
    std::fs::write(benchmark_file, json_data)?;
    
    // Generate summary report
    generate_benchmark_report(&results, &output).await?;
    
    Ok(())
}

async fn analyze_results(pattern: String) -> Result<()> {
    println!("📊 Analyzing results matching pattern: {}", pattern);
    
    let mut analysis = AnalysisPipeline::new();
    
    analysis
        .load_experiment_results(&pattern)?
        .statistical_analysis()?
        .regression_modeling()?
        .visualization("analysis_plots")?
        .report_generation()?;
    
    analysis.execute().await?;
    
    println!("✅ Analysis completed");
    Ok(())
}

async fn generate_example_config(output: PathBuf) -> Result<()> {
    println!("📝 Generating example experiment configuration");
    
    let example_config = r#"{
  "experiment": {
    "name": "Example Scaling Experiment",
    "description": "Demonstrates scaling analysis across WE3 frameworks",
    "frameworks": ["GeneratingFunctions", "TensorNetworks"],
    "parameters": {
      "problem_size": [100, 1000, 10000, 100000],
      "precision": [1e-6, 1e-9, 1e-12],
      "algorithm_variant": ["fast", "accurate"]
    },
    "repetitions": 50,
    "design_type": "FullFactorial"
  },
  "resources": {
    "max_memory_gb": 16,
    "max_cpu_cores": 8,
    "timeout_minutes": 60
  },
  "analysis": {
    "confidence_level": 0.95,
    "generate_plots": true,
    "regression_models": ["linear", "polynomial", "exponential"]
  }
}"#;
    
    let config_file = output.join("example_experiment.json");
    std::fs::write(config_file, example_config)?;
    
    println!("✅ Example configuration saved to: {}", output.join("example_experiment.json").display());
    
    Ok(())
}

async fn generate_benchmark_report(results: &[BenchmarkResult], output: &PathBuf) -> Result<()> {
    println!("📋 Generating benchmark report");
    
    let mut report = String::new();
    report.push_str("# WE3 Framework Benchmark Report\n\n");
    report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- Total benchmarks: {}\n", results.len()));
    
    let successful = results.iter().filter(|r| r.statistics.successful_runs > 0).count();
    report.push_str(&format!("- Successful benchmarks: {}\n", successful));
    
    let total_runs: usize = results.iter().map(|r| r.statistics.successful_runs).sum();
    report.push_str(&format!("- Total experimental runs: {}\n\n", total_runs));
    
    report.push_str("## Framework Performance Summary\n\n");
    
    for result in results {
        report.push_str(&format!("### {}\n", result.benchmark.name));
        report.push_str(&format!("- Framework: {:?}\n", result.benchmark.framework));
        report.push_str(&format!("- Category: {:?}\n", result.benchmark.category));
        report.push_str(&format!("- Runs: {}\n", result.statistics.successful_runs));
        report.push_str(&format!("- Mean execution time: {:.3}ms\n", 
                                result.statistics.execution_time_stats.mean * 1000.0));
        report.push_str(&format!("- Peak memory: {:.1}MB\n", 
                                result.statistics.memory_stats.max / 1024.0 / 1024.0));
        
        if let Some(throughput) = result.metrics.throughput {
            report.push_str(&format!("- Throughput: {:.1} ops/sec\n", throughput));
        }
        
        report.push_str("\n");
    }
    
    report.push_str("## Detailed Analysis\n\n");
    report.push_str("For detailed analysis including statistical summaries, scaling plots, and comparative analysis, ");
    report.push_str("see the generated data files and visualizations in the output directory.\n");
    
    let report_file = output.join("benchmark_report.md");
    std::fs::write(report_file, report)?;
    
    println!("✅ Benchmark report saved");
    Ok(())
}