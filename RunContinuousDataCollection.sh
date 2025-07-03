#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LEADERBOARD_ROOT="$SCRIPT_DIR/Leaderboard"
export TEAM_CODE_ROOT="$SCRIPT_DIR/agents"

export PYTHONPATH="$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$PYTHONPATH"

export TEAM_AGENT="$SCRIPT_DIR/data_collection_agent.py"

# Set up cleanup trap
trap 'echo "🧹 Cleaning up temporary files..."; rm -f "$SCRIPT_DIR"/temp_mission_*.xml; exit' INT TERM EXIT

export MISSIONS="$LEADERBOARD_ROOT/data/missions_training.xml"
export MISSIONS_SUBSET="0"



export REPETITIONS="1"

export RECORD=
export RECORD_CONTROL=
export RESUME=0

export QUALIFIER=
export EVALUATION=
export DEVELOPMENT=1

# Base seed - will be randomized for each mission
# Each mission gets a unique seed: timestamp + base_seed + attempt_number
export BASE_SEED=0

# Available maps and presets for maximum diversity
# Each map has specific preset ranges
AVAILABLE_MAPS=("Moon_Map_01" "Moon_Map_02")

# Map-specific preset ranges
MOON_MAP_01_PRESETS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
MOON_MAP_02_PRESETS=("11" "12" "13")

# Function to create a dynamic mission file for specific map/preset
create_mission_file() {
    local map_name="$1"
    local preset_id="$2"
    
    # Create temporary mission file
    local temp_mission_file="$SCRIPT_DIR/temp_mission_${map_name}_${preset_id}.xml"
    
    cat > "$temp_mission_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<missions>
  <mission id="0" map="$map_name" preset="$preset_id"/>
</missions>
EOF
    
    # Update the MISSIONS environment variable to use our dynamic file
    export MISSIONS="$temp_mission_file"
    
    echo "📄 Created mission file: $temp_mission_file"
    echo "   Map: $map_name, Preset: $preset_id"
}

# Function to check if we should continue running
should_continue() {
    # Check if data collection directory exists and has enough data
    if [ -d "$SCRIPT_DIR/data_collection" ]; then
        # Count total trajectories collected across ALL mission folders
        trajectory_count=$(find "$SCRIPT_DIR/data_collection" -name "trajectory_*.npz" | wc -l)
        
        # Check if we have a summary file to get total time
        if [ -f "$SCRIPT_DIR/data_collection/collection_summary.json" ]; then
            # Extract total mission duration from summary (if available)
            # For now, we'll just check trajectory count as a simple metric
            echo "📊 Current trajectory count: $trajectory_count"
            
            # Continue if we have less than 1000 trajectories (adjust as needed)
            if [ $trajectory_count -lt 1000 ]; then
                return 0  # Continue
            else
                echo "🎯 Target trajectory count reached! Stopping continuous collection."
                return 1  # Stop
            fi
        else
            # No summary file yet, continue
            return 0
        fi
    else
        # No data collection directory yet, continue
        return 0
    fi
}

# Function to run a single leaderboard execution
run_leaderboard() {
    # Create mission-specific data directory
    mission_data_dir="$SCRIPT_DIR/data_collection/mission_$1"
    export MISSION_DATA_DIR="$mission_data_dir"
    
    # Generate random seed for this mission using timestamp and attempt number
    timestamp=$(date +%s)
    mission_seed=$((timestamp + BASE_SEED + $1))
    export SEED="$mission_seed"
    
    # Select map and preset for this mission (both random)
    # Use the mission seed to ensure reproducible randomness for this attempt
    RANDOM=$mission_seed
    map_index=$((RANDOM % ${#AVAILABLE_MAPS[@]}))
    map_name="${AVAILABLE_MAPS[$map_index]}"
    
    # Generate a new random number for preset selection based on selected map
    preset_seed=$((mission_seed + 1000))  # Offset to get different random sequence
    RANDOM=$preset_seed
    
    if [ "$map_name" = "Moon_Map_01" ]; then
        preset_index=$((RANDOM % ${#MOON_MAP_01_PRESETS[@]}))
        preset_id="${MOON_MAP_01_PRESETS[$preset_index]}"
    elif [ "$map_name" = "Moon_Map_02" ]; then
        preset_index=$((RANDOM % ${#MOON_MAP_02_PRESETS[@]}))
        preset_id="${MOON_MAP_02_PRESETS[$preset_index]}"
    else
        echo "❌ Unknown map: $map_name"
        exit 1
    fi
    
    echo "🚀 Starting leaderboard execution (attempt $1)..."
    echo "🔄 Resume disabled: $RESUME"
    echo "📁 Mission data directory: $mission_data_dir"
    echo "🎲 Using random seed: $mission_seed"
    echo "🗺️  Using map: $map_name with preset: $preset_id"
    
    # Create dynamic mission file for this specific map/preset combination
    create_mission_file "$map_name" "$preset_id"
    
    # Capture output to detect failures
    output=$(python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
      --missions="${MISSIONS}" \
      --missions-subset="${MISSIONS_SUBSET}" \
      --seed="${SEED}" \
      --repetitions="${REPETITIONS}" \
      --agent="${TEAM_AGENT}" \
      --agent-config="${TEAM_CONFIG}" \
      --record="${RECORD}" \
      --record-control="${RECORD_CONTROL}" \
      --resume="${RESUME}" \
      --qualifier="${QUALIFIER}" \
      --evaluation="${EVALUATION}" \
      --development="${DEVELOPMENT}" 2>&1)
    
    exit_code=$?
    echo "$output"
    echo "📊 Leaderboard execution completed with exit code: $exit_code"
    
    # Check if the output indicates a failure that should trigger restart
    if echo "$output" | grep -q "AgentRuntimeError\|Simulation_crash\|Agent_setup_failure\|Invalid_sensors"; then
        echo "🚨 Failure detected in output - mission needs restart"
        return 1  # Indicate failure
    elif echo "$output" | grep -q "Target collection time reached\|mission_complete\|Data collection complete"; then
        echo "✅ Normal completion detected - mission finished successfully"
        return 0  # Indicate success
    else
        # Default to exit code behavior
        return $exit_code
    fi
}

# Main continuous execution loop
echo "🔄 Starting continuous data collection..."
echo "📁 Data will be saved to: $SCRIPT_DIR/data_collection"
echo "⏹️  Press Ctrl+C to stop at any time"
echo ""

attempt=1
while should_continue; do
    echo "🔄 ========================================="
    echo "🔄 Continuous Data Collection - Attempt $attempt"
    echo "🔄 ========================================="
    
    # Run the leaderboard
    run_leaderboard $attempt
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Leaderboard completed successfully!"
        echo "🎯 Mission completed normally - stopping continuous collection."
        break
    else
        echo "⚠️  Leaderboard detected failure (exit code: $exit_code)"
        echo "🔄 This is expected for collision/boundary violations - restarting..."
        
        # Initialize simulator before restarting
        echo "🚀 Initializing Lunar Simulator before restart..."
        bash "$SCRIPT_DIR/RunLunarSimulator.sh" &
        simulator_pid=$!
        
        # Wait a bit for simulator to start up
        echo "⏳ Waiting for simulator to initialize..."
        sleep 10
        
        # Kill the simulator process (it will be restarted by the leaderboard)
        if kill -0 $simulator_pid 2>/dev/null; then
            echo "🛑 Stopping simulator process before restart..."
            kill $simulator_pid
            wait $simulator_pid 2>/dev/null
        fi
        
        # Small delay before restarting
        echo "⏳ Waiting 5 seconds before restart..."
        sleep 5
        
        attempt=$((attempt + 1))
    fi
done

echo ""
echo "🏁 Continuous data collection finished!"
echo "📊 Final statistics:"
if [ -d "$SCRIPT_DIR/data_collection" ]; then
    trajectory_count=$(find "$SCRIPT_DIR/data_collection" -name "trajectory_*.npz" | wc -l)
    echo "   Total trajectories collected: $trajectory_count"
    
    # List all mission directories
    mission_dirs=$(find "$SCRIPT_DIR/data_collection" -type d -name "mission_*" | sort)
    if [ -n "$mission_dirs" ]; then
        echo "   Mission directories:"
        for dir in $mission_dirs; do
            mission_trajectories=$(find "$dir" -name "trajectory_*.npz" | wc -l)
            echo "     $(basename "$dir"): $mission_trajectories trajectories"
        done
    fi
    
    if [ -f "$SCRIPT_DIR/data_collection/collection_summary.json" ]; then
        echo "   Collection summary: $SCRIPT_DIR/data_collection/collection_summary.json"
    fi
fi
echo "📁 Data directory: $SCRIPT_DIR/data_collection"

# Clean up temporary mission files
echo "🧹 Cleaning up temporary mission files..."
rm -f "$SCRIPT_DIR"/temp_mission_*.xml 