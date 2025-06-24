import pandas as pd
import faker
import uuid
import datetime
import json

# Instantiate Faker
fake = faker.Faker()

# Part 1: Setup and Constants
AGENT_SPECS = {
    "agent_name": "MySuperAgent",
    "agent_version": "v2.1.3",
    "agent_architecture_type": "RAG_ENHANCED",
    "base_llm_model": "openai/gpt-4o-2024-11-20",
    "base_vlm_model": "anthropic/claude-3-7-sonnet-20250219",
    "agent_tools_json": json.dumps(["web_search", "code_interpreter", "vlm_analyze", "file_io"])
}

BENCHMARK_CONTEXT = {
    "benchmark_name": "GAIA",
    "benchmark_version": "2025.Q4",
    "task_id": "gaia_task_231",
    "task_prompt": "Examine the attached image of a cookie recipe...",  # Truncated for brevity
    "task_difficulty_level": 2,
    "environment_type": "OS",
    "environment_os": "UBUNTU",
    "optimal_steps_human": 6
}

# Part 2: Scenario Functions
def create_efficient_success_run():
    """Generate a dictionary for a successful run."""
    return {
        "run_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "run_timestamp_utc": fake.iso8601(),
        "agent_name": AGENT_SPECS["agent_name"],
        "agent_version": AGENT_SPECS["agent_version"],
        "agent_architecture_type": AGENT_SPECS["agent_architecture_type"],
        "base_llm_model": AGENT_SPECS["base_llm_model"],
        "base_vlm_model": AGENT_SPECS["base_vlm_model"],
        "agent_tools_json": AGENT_SPECS["agent_tools_json"],
        "benchmark_name": BENCHMARK_CONTEXT["benchmark_name"],
        "benchmark_version": BENCHMARK_CONTEXT["benchmark_version"],
        "task_id": BENCHMARK_CONTEXT["task_id"],
        "task_prompt": BENCHMARK_CONTEXT["task_prompt"],
        "task_difficulty_level": BENCHMARK_CONTEXT["task_difficulty_level"],
        "environment_type": BENCHMARK_CONTEXT["environment_type"],
        "environment_os": BENCHMARK_CONTEXT["environment_os"],
        "optimal_steps_human": BENCHMARK_CONTEXT["optimal_steps_human"],
        "final_outcome": "SUCCESS",
        "outcome_reason": "Correct file content generated",
        "total_steps": 5,
        "step_efficiency_ratio": 0.83,
        "estimated_cost_usd": 0.08,
        "total_llm_calls": 3,
        "total_tool_calls": 4,
        "trajectory_full_json": json.dumps([
            {"step_index": 0, "thought": "I need to analyze the image to get the recipe details.", "action_type": "VLM_ANALYZE", "action_input": "image.jpg", "observation": "Recipe: 'Magic Cookies', Prep Time: '45 minutes', Ratings: '1250'"},
            {"step_index": 1, "thought": "Now I need the price of eggs in San Francisco.", "action_type": "WEB_SEARCH", "action_input": "current price of a dozen eggs in San Francisco CA", "observation": "The price is $6.24 per dozen at Safeway."},
            {"step_index": 2, "thought": "I need to calculate the cost per egg.", "action_type": "CODE_INTERPRETER", "action_input": "print(6.24 / 12)", "observation": "0.52"},
            {"step_index": 3, "thought": "I have all the information. I will write it to the specified file.", "action_type": "FILE_IO", "action_input": {"operation": "write", "filename": "recipe_analysis.txt", "content": "recipe_name: Magic Cookies\\nprep_time: 45 minutes\\nrating_count: 1250\\ncost_per_egg: 0.52"}, "observation": "File written successfully."},
            {"step_index": 4, "thought": "Task complete.", "action_type": "FINISH", "action_input": "Success", "observation": ""}
        ])
    }

def create_inefficient_success_run():
    """Generate a dictionary for an inefficient but successful run."""
    return {
        "run_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "run_timestamp_utc": fake.iso8601(),
        "agent_name": AGENT_SPECS["agent_name"],
        "agent_version": AGENT_SPECS["agent_version"],
        "agent_architecture_type": AGENT_SPECS["agent_architecture_type"],
        "base_llm_model": AGENT_SPECS["base_llm_model"],
        "base_vlm_model": AGENT_SPECS["base_vlm_model"],
        "agent_tools_json": AGENT_SPECS["agent_tools_json"],
        "benchmark_name": BENCHMARK_CONTEXT["benchmark_name"],
        "benchmark_version": BENCHMARK_CONTEXT["benchmark_version"],
        "task_id": BENCHMARK_CONTEXT["task_id"],
        "task_prompt": BENCHMARK_CONTEXT["task_prompt"],
        "task_difficulty_level": BENCHMARK_CONTEXT["task_difficulty_level"],
        "environment_type": BENCHMARK_CONTEXT["environment_type"],
        "environment_os": BENCHMARK_CONTEXT["environment_os"],
        "optimal_steps_human": BENCHMARK_CONTEXT["optimal_steps_human"],
        "final_outcome": "SUCCESS",
        "outcome_reason": "Correct file content generated despite extra steps",
        "total_steps": 8,
        "step_efficiency_ratio": 1.33,
        "estimated_cost_usd": 0.15,
        "total_llm_calls": 5,
        "total_tool_calls": 6,
        "trajectory_full_json": json.dumps([
            {"step_index": 0, "thought": "First, I'll get the recipe name from the image.", "action_type": "VLM_ANALYZE", "action_input": {"image": "image.jpg", "query": "recipe name"}, "observation": "Recipe: 'Magic Cookies'"},
            {"step_index": 1, "thought": "Now I need the prep time.", "action_type": "VLM_ANALYZE", "action_input": {"image": "image.jpg", "query": "preparation time"}, "observation": "Prep Time: '45 minutes'"},
            {"step_index": 2, "thought": "Now for the ratings.", "action_type": "VLM_ANALYZE", "action_input": {"image": "image.jpg", "query": "number of ratings"}, "observation": "Ratings: '1250'"},
            {"step_index": 3, "thought": "I'll search for egg prices.", "action_type": "WEB_SEARCH", "action_input": "egg prices", "observation": "General information about egg prices."},
            {"step_index": 4, "thought": "That wasn't specific enough. I need to search again.", "action_type": "WEB_SEARCH", "action_input": "price dozen eggs San Francisco", "observation": "The price is $6.24 per dozen."},
            {"step_index": 5, "thought": "I need to calculate the cost per egg.", "action_type": "CODE_INTERPRETER", "action_input": "print(6.24 / 12)", "observation": "0.52"},
            {"step_index": 6, "thought": "Now I will write the data to the file.", "action_type": "FILE_IO", "action_input": {"operation": "write", "filename": "recipe_analysis.txt", "content": "recipe_name: Magic Cookies\\nprep_time: 45 minutes\\nrating_count: 1250\\ncost_per_egg: 0.52"}, "observation": "File written successfully."},
            {"step_index": 7, "thought": "Task complete.", "action_type": "FINISH", "action_input": "Success", "observation": ""}
        ])
    }

def create_plausible_failure_run():
    """Generate a dictionary for a failed run."""
    return {
        "run_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "run_timestamp_utc": fake.iso8601(),
        "agent_name": AGENT_SPECS["agent_name"],
        "agent_version": AGENT_SPECS["agent_version"],
        "agent_architecture_type": AGENT_SPECS["agent_architecture_type"],
        "base_llm_model": AGENT_SPECS["base_llm_model"],
        "base_vlm_model": AGENT_SPECS["base_vlm_model"],
        "agent_tools_json": AGENT_SPECS["agent_tools_json"],
        "benchmark_name": BENCHMARK_CONTEXT["benchmark_name"],
        "benchmark_version": BENCHMARK_CONTEXT["benchmark_version"],
        "task_id": BENCHMARK_CONTEXT["task_id"],
        "task_prompt": BENCHMARK_CONTEXT["task_prompt"],
        "task_difficulty_level": BENCHMARK_CONTEXT["task_difficulty_level"],
        "environment_type": BENCHMARK_CONTEXT["environment_type"],
        "environment_os": BENCHMARK_CONTEXT["environment_os"],
        "optimal_steps_human": BENCHMARK_CONTEXT["optimal_steps_human"],
        "final_outcome": "FAILURE",
        "outcome_reason": "Incorrect file content",
        "step_efficiency_ratio": None,
        "total_llm_calls": None,
        "total_tool_calls": None,
        "failure_step_index": 5,
        "failure_type": "DATA_EXTRACTION_ERROR",
        "error_message": "Evaluation failed: 'recipe_analysis.txt' content mismatch. Expected rating count '1250', found '4.5 stars'.",
        "total_steps": 6,
        "estimated_cost_usd": 0.09,
        "trajectory_full_json": json.dumps([
            {"step_index": 0, "thought": "I need to analyze the image to get the recipe details.", "action_type": "VLM_ANALYZE", "action_input": "image.jpg", "observation": "Recipe: 'Magic Cookies', Prep Time: '45 minutes', Ratings: '4.5 stars'"},
            {"step_index": 1, "thought": "It seems I have all the info from the image. Now I need the price of eggs.", "action_type": "WEB_SEARCH", "action_input": "price of a dozen eggs in San Francisco CA", "observation": "The price is $6.24 per dozen."},
            {"step_index": 2, "thought": "I need to calculate the cost per egg.", "action_type": "CODE_INTERPRETER", "action_input": "print(6.24 / 12)", "observation": "0.52"},
            {"step_index": 3, "thought": "I will now write all the extracted information to the specified file.", "action_type": "FILE_IO", "action_input": {"operation": "write", "filename": "recipe_analysis.txt", "content": "recipe_name: Magic Cookies\\nprep_time: 45 minutes\\nrating_count: 4.5 stars\\ncost_per_egg: 0.52"}, "observation": "File written successfully."},
            {"step_index": 4, "thought": "Task seems complete. I will finish.", "action_type": "FINISH", "action_input": "Success (Mistakenly)", "observation": ""},
            {"step_index": 5, "thought": None, "action_type": "EVALUATOR_CHECK", "action_input": "recipe_analysis.txt", "observation": "Content mismatch failure"}
        ])
    }

# Part 3: Main Execution Logic
def main():
    """Main function to generate and save the dataset."""
    # Initialize empty list for all runs
    all_runs = []
    
    # Generate the data by calling the functions
    # Call create_efficient_success_run() twice
    all_runs.append(create_efficient_success_run())
    all_runs.append(create_efficient_success_run())
    
    # Call create_inefficient_success_run() once
    all_runs.append(create_inefficient_success_run())
    
    # Call create_plausible_failure_run() twice
    all_runs.append(create_plausible_failure_run())
    all_runs.append(create_plausible_failure_run())
    
    # Convert the all_runs list into a pandas DataFrame
    df = pd.DataFrame(all_runs)
    
    # Save the DataFrame to data/simulated_hud_evals_advanced.csv
    df.to_csv("data/simulated_hud_evals_advanced.csv", index=False)
    
    print(f"Generated dataset with {len(df)} rows")
    print(f"Dataset saved to: data/simulated_hud_evals_advanced.csv")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main() 