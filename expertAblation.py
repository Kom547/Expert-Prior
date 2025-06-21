import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (modify path if needed)
csv_path = "evaluation_result/MoE_test_all_1.csv"
df = pd.read_csv(csv_path, sep=",")


# Parse the `expert_attack` column to extract experimental variables
def parse_expert_attack(path):
    filename = path.rsplit("/", 1)[-1]
    parts = filename.split("_")
    low_quality_removed = parts[0] == "f"
    balanced = "nbal" not in parts
    try:
        ratio_value = float(parts[-1])
    except ValueError:
        ratio_value = float("nan")
    return low_quality_removed, balanced, ratio_value


parsed = df["expert_attack"].apply(parse_expert_attack)
df[["low_quality_removed_bool", "balanced_bool", "ratio"]] = pd.DataFrame(parsed.tolist(), index=df.index)
df["low_quality_removed"] = df["low_quality_removed_bool"].map({True: "removed", False: "not_removed"})
df["balanced"] = df["balanced_bool"].map({True: "balanced", False: "not_balanced"})

# Create a column for the expert type (combination of removal + balanced)
df["expert_type"] = df["low_quality_removed"] + " & " + df["balanced"]

# Identify unique (env_name, algo) combinations and take up to six
env_algo_groups = df[["env_name", "algo"]].drop_duplicates()
selected_groups = env_algo_groups.head(6).values.tolist()

# Create a pivot table for all combinations of env_name, algo, ratio, and expert_type
pivot_table = df.pivot_table(
    index=["env_name", "algo", "ratio"],
    columns="expert_type",
    values="collision_rate",
    aggfunc="mean"
).reset_index()

# Sort rows for readability
pivot_table = pivot_table.sort_values(by=["env_name", "algo", "ratio"])

# Save the combined pivot table to a single CSV
output_filename = "evaluation_result/all_env_algo_expert_collision_rate.csv"
pivot_table.to_csv(output_filename, index=False)

# Display a confirmation
print(f"Saved combined CSV as: {output_filename}")

# Set up a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
fig.suptitle("各环境 (env_name + algo) 下不同专家类型的平均碰撞率 (collision_rate)", fontsize=16)

for idx, (env, algo) in enumerate(selected_groups):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    subset = df[(df["env_name"] == env) & (df["algo"] == algo)]
    # Group by expert_type and ratio, then compute mean collision_rate
    grouped = subset.groupby(["expert_type", "ratio"])["collision_rate"].mean().reset_index()

    # Plot each expert type as a separate line
    for expert_type, group_df in grouped.groupby("expert_type"):
        ax.plot(group_df["ratio"], group_df["collision_rate"], marker="o", label=expert_type)

    ax.set_xscale("log")
    ax.set_title(f"{env}  |  {algo}")
    ax.set_xlabel("下采样比率 (ratio)")
    ax.set_ylabel("平均碰撞率 (collision_rate)")
    ax.legend(fontsize="small")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
