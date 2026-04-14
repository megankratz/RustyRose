import json
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_rust_benchmarks():
    print("Running Rust benchmarks...")
    subprocess.run(["cargo", "bench", "--bench", "feature_benchmarks"], check=True)

def parse_rust_results():
    results = {}
    criterion_dir = "target/criterion"
    if not os.path.exists(criterion_dir):
        return results
    
    for root, dirs, files in os.walk(criterion_dir):
        if "estimates.json" in files and root.endswith("new"):
            # root is target/criterion/group/sub/param/new
            rel = os.path.relpath(root, criterion_dir)
            parts = rel.split(os.sep)
            # parts is [group, ..., "new"]
            if len(parts) >= 3:
                group = parts[0]
                if group == "report": continue
                param = "/".join(parts[1:-1])
                
                with open(os.path.join(root, "estimates.json"), "r") as f:
                    data = json.load(f)
                    if group not in results: results[group] = {}
                    results[group][param] = data["mean"]["point_estimate"]
    return results

def run_python_benchmarks():
    print("Running Python benchmarks...")
    python_cmd = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python3"
    subprocess.run([python_cmd, "scripts/librosa_benchmark.py"], check=True)

def parse_python_results():
    with open("benchmark_results/librosa_results.json", "r") as f:
        return json.load(f)

def plot_results(rust_res, py_res):
    # (Removed Frame-based Jonathan features)

    # (Removed frequency conversion plotting)
    # Feature categories and their target plot files
    categories = {
        "time_domain": "plot_time_domain.png",
        "spectral_scaling": "plot_spectral_scaling.png",
        "spectral_features": "plot_spectral_features.png",
        "utils_effects": "plot_utils_effects.png"
    }
    
    for group_id, filename in categories.items():
        if group_id not in rust_res: continue
        
        data = []
        # Support both frame-based and single-value naming
        for key in rust_res[group_id]:
            data.append({
                "Function": key,
                "Library": "RustyRose (Rust)",
                "Time (us)": rust_res[group_id][key] / 1e3
            })
        if group_id in py_res:
             for key in py_res[group_id]:
                data.append({
                    "Function": key,
                    "Library": "Librosa (Python)",
                    "Time (us)": py_res[group_id][key] / 1e3
                })
        
        if data:
            df = pd.DataFrame(data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Function", y="Time (us)", hue="Library", palette="muted")
            plt.title(f"{group_id.replace('_', ' ').title()} Performance")
            plt.ylabel("Time (microseconds)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"benchmark_results/{filename}")
            plt.close()

    print("Separate category plots saved to benchmark_results/")

    print("Plots saved to benchmark_results/")

def generate_markdown_report(rust_res, py_res):
    report = "# Performance Comparison: RustyRose vs Librosa\n\n"
    # (Removed frequency conversions)

    # Feature Category Report
    feature_groups = {
        "time_domain": "Time-Domain Processing",
        "spectral_scaling": "Spectral Scaling & RMS",
        "spectral_features": "Spectral Features (Tonnetz/Poly)",
        "utils_effects": "Utilities & Audio Effects"
    }
    
    for group_id, group_name in feature_groups.items():
        if group_id not in rust_res: continue
        report += f"\n## {group_name}\n\n"
        report += "| Function | RustyRose (us) | Librosa (us) | Speedup |\n"
        report += "| :--- | :--- | :--- | :--- |\n"
        
        rust_group = rust_res[group_id]
        py_group = py_res.get(group_id, {})
        
        for func in sorted(rust_group.keys()):
            r_time = rust_group[func] / 1e3
            p_time = py_group.get(func, 0) / 1e3
            speedup = p_time / r_time if r_time > 0 else 0
            report += f"| {func} | {r_time:.2f} | {p_time:.2f} | {speedup:.2f}x |\n"
            
    with open("benchmark_results/report.md", "w") as f:
        f.write(report)
    print("Markdown report saved to benchmark_results/report.md")

if __name__ == "__main__":
    run_rust_benchmarks()
    run_python_benchmarks()
    
    rust_results = parse_rust_results()
    python_results = parse_python_results()
    
    generate_markdown_report(rust_results, python_results)
    plot_results(rust_results, python_results)
