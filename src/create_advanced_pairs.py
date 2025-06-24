import pandas as pd

def main():
    """Create pairwise preferences dataset for reward model training."""
    
    # 1. Import pandas (already done at top)
    # 2. Read the simulated_hud_evals_advanced.csv file
    print("Reading dataset...")
    df = pd.read_csv('data/simulated_hud_evals_advanced.csv')
    
    # 3. Find all successful runs and store their trajectory_full_json values
    print("Extracting successful trajectories...")
    successful_runs = df[df['final_outcome'] == 'SUCCESS']
    successful_trajectories = successful_runs['trajectory_full_json'].tolist()
    print(f"Found {len(successful_trajectories)} successful trajectories")
    
    # 4. Find all failed runs and store their trajectory_full_json values
    print("Extracting failed trajectories...")
    failed_runs = df[df['final_outcome'] == 'FAILURE']
    failed_trajectories = failed_runs['trajectory_full_json'].tolist()
    print(f"Found {len(failed_trajectories)} failed trajectories")
    
    # 5. Create all possible pairs where 'chosen' is successful and 'rejected' is failed
    print("Creating pairwise preferences...")
    pairs = []
    for chosen_trajectory in successful_trajectories:
        for rejected_trajectory in failed_trajectories:
            pairs.append({
                'chosen': chosen_trajectory,
                'rejected': rejected_trajectory
            })
    
    print(f"Created {len(pairs)} pairwise preference pairs")
    
    # 6. Convert pairs list to DataFrame with 'chosen' and 'rejected' columns
    print("Converting to DataFrame...")
    pairs_df = pd.DataFrame(pairs)
    
    # 7. Save DataFrame to data/pairwise_preferences_advanced.csv without index
    print("Saving pairwise preferences dataset...")
    pairs_df.to_csv('data/pairwise_preferences_advanced.csv', index=False)
    
    print(f"✅ Successfully created pairwise preferences dataset!")
    print(f"📁 Saved to: data/pairwise_preferences_advanced.csv")
    print(f"📊 Dataset shape: {pairs_df.shape}")
    print(f"📋 Columns: {list(pairs_df.columns)}")
    
    # Show sample of the data
    print("\n📝 Sample of pairwise preferences:")
    print(pairs_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main() 