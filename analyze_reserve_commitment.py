import pandas as pd
import os

def load_generator_data():
    """Load generator characteristics from the data directory"""
    base_dir = "/Users/tommie/Documents/thesis/project/data"
    generators_df = pd.read_csv(os.path.join(base_dir, "Generators.csv"))
    generators_df.set_index('Generator', inplace=True)
    return generators_df

def analyze_reserve_commitment(csv_path, generators_df):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by Generator and Hour to get unique commitment states
    commitment_df = df.groupby(['Generator', 'Hour'])['wValue'].first().reset_index()
    
    # For each committed generator-hour combination, check if any reserves are provided
    committed_gens = commitment_df[commitment_df['wValue'] == 1.0]
    
    # Get the total number of committed generator-hours
    total_committed = len(committed_gens)
    
    # For each committed generator-hour, check if any reserves are provided
    no_reserves_count = 0
    no_reserves_details = []
    reserves_by_gen = {}  # Track reserve provision by generator
    
    for _, row in committed_gens.iterrows():
        gen = row['Generator']
        hour = row['Hour']
        
        # Get all reserve values for this generator-hour
        reserves = df[(df['Generator'] == gen) & 
                     (df['Hour'] == hour) & 
                     (df['wValue'] == 1.0)]['s_res_Value']
        
        # Track reserve provision by generator
        if gen not in reserves_by_gen:
            reserves_by_gen[gen] = {
                'total_hours': 0, 
                'hours_with_reserves': 0,
                'total_reserves_provided': 0,
                'min_up_time': generators_df.loc[gen, 'UT'] if gen in generators_df.index else 0,
                'no_load_cost': generators_df.loc[gen, 'NoLoadConsumption'] if gen in generators_df.index else 0,
                'startup_cost': generators_df.loc[gen, 'StartupCost'] if gen in generators_df.index else 0,
                'p_min': generators_df.loc[gen, 'MinRunCapacity'] if gen in generators_df.index else 0,
                'p_max': generators_df.loc[gen, 'MaxRunCapacity'] if gen in generators_df.index else 0
            }
        reserves_by_gen[gen]['total_hours'] += 1
        if reserves.sum() > 0:
            reserves_by_gen[gen]['hours_with_reserves'] += 1
            reserves_by_gen[gen]['total_reserves_provided'] += reserves.sum()
        
        # If all reserve values are 0, this is a case of commitment without reserves
        if reserves.sum() == 0:
            no_reserves_count += 1
            no_reserves_details.append({
                'Generator': gen,
                'Hour': hour,
                'Total_Reserves': 0,
                'Min_Up_Time': generators_df.loc[gen, 'UT'] if gen in generators_df.index else 0,
                'No_Load_Cost': generators_df.loc[gen, 'NoLoadConsumption'] if gen in generators_df.index else 0,
                'Startup_Cost': generators_df.loc[gen, 'StartupCost'] if gen in generators_df.index else 0,
                'P_min': generators_df.loc[gen, 'MinRunCapacity'] if gen in generators_df.index else 0
            })
    
    # Calculate statistics
    stats = {
        'Total_Committed_Generator_Hours': total_committed,
        'Committed_Without_Reserves': no_reserves_count,
        'Percentage_Without_Reserves': (no_reserves_count / total_committed * 100) if total_committed > 0 else 0
    }
    
    # Create detailed DataFrame for cases without reserves
    details_df = pd.DataFrame(no_reserves_details)
    
    # Analyze generator characteristics
    gen_analysis = []
    for gen, data in reserves_by_gen.items():
        gen_info = generators_df[generators_df.index == gen].iloc[0] if gen in generators_df.index else None
        if gen_info is not None:
            gen_analysis.append({
                'Generator': gen,
                'Total_Hours_Committed': data['total_hours'],
                'Hours_With_Reserves': data['hours_with_reserves'],
                'Hours_Without_Reserves': data['total_hours'] - data['hours_with_reserves'],
                'Percentage_With_Reserves': (data['hours_with_reserves'] / data['total_hours'] * 100) if data['total_hours'] > 0 else 0,
                'Total_Reserves_Provided': data['total_reserves_provided'],
                'Avg_Reserves_When_Provided': data['total_reserves_provided'] / data['hours_with_reserves'] if data['hours_with_reserves'] > 0 else 0,
                'GeneratorType': gen_info.get('GeneratorType', 'Unknown'),
                'P_min': gen_info.get('MinRunCapacity', 0),
                'P_max': gen_info.get('MaxRunCapacity', 0),
                'R_max': gen_info.get('RampUp', 0),
                'R_min': gen_info.get('RampDown', 0),
                'UT_g': gen_info.get('UT', 0),
                'DT_g': gen_info.get('DT', 0),
                'K': gen_info.get('NoLoadConsumption', 0),  # No-load cost
                'S': gen_info.get('StartupCost', 0),  # Startup cost
                'Total_No_Load_Cost': data['total_hours'] * gen_info.get('NoLoadConsumption', 0),
                'Potential_Startup_Cost': data['total_hours'] * gen_info.get('StartupCost', 0) / 24  # Assuming one startup per day
            })
    
    gen_analysis_df = pd.DataFrame(gen_analysis)
    
    return stats, details_df, gen_analysis_df

def main():
    base_dir = "/Users/tommie/Documents/thesis/project/results/sequential"
    seasons = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]
    
    # Load generator data
    generators_df = load_generator_data()
    
    print("\nAnalyzing Reserve Commitment Patterns\n")
    print("=" * 80)
    
    for season in seasons:
        csv_path = os.path.join(base_dir, f"reserve_solution_{season}.csv")
        print(f"\nSeason: {season}")
        print("-" * 40)
        
        stats, details, gen_analysis = analyze_reserve_commitment(csv_path, generators_df)
        
        print(f"Total Committed Generator-Hours: {stats['Total_Committed_Generator_Hours']}")
        print(f"Cases of Commitment Without Reserves: {stats['Committed_Without_Reserves']}")
        print(f"Percentage Without Reserves: {stats['Percentage_Without_Reserves']:.2f}%")
        
        if not gen_analysis.empty:
            print("\nGenerator Characteristics Analysis:")
            
            # Analyze generators that are committed but not providing reserves
            no_reserves_gens = gen_analysis[gen_analysis['Hours_With_Reserves'] == 0]
            if not no_reserves_gens.empty:
                print("\nGenerators Never Providing Reserves:")
                print("\nCost Analysis for Generators Never Providing Reserves:")
                cost_analysis = no_reserves_gens[['Generator', 'GeneratorType', 'Total_Hours_Committed', 
                                                'K', 'S', 'Total_No_Load_Cost', 'Potential_Startup_Cost',
                                                'P_min', 'P_max', 'UT_g']]
                print(cost_analysis.to_string())
                
                print("\nTotal Additional Costs from Unused Commitments:")
                total_no_load = no_reserves_gens['Total_No_Load_Cost'].sum()
                total_startup = no_reserves_gens['Potential_Startup_Cost'].sum()
                print(f"Total No-Load Costs: {total_no_load:.2f}")
                print(f"Total Potential Startup Costs: {total_startup:.2f}")
                print(f"Total Additional Costs: {total_no_load + total_startup:.2f}")
            
            # Analyze generators that are providing reserves
            reserves_gens = gen_analysis[gen_analysis['Hours_With_Reserves'] > 0]
            if not reserves_gens.empty:
                print("\nGenerators Providing Reserves:")
                print("\nReserve Provision Analysis:")
                reserve_analysis = reserves_gens[['Generator', 'GeneratorType', 'Total_Hours_Committed',
                                               'Hours_With_Reserves', 'Total_Reserves_Provided',
                                               'Avg_Reserves_When_Provided', 'P_min', 'P_max']]
                print(reserve_analysis.to_string())
            
            print("\nSummary by Generator Type:")
            type_summary = gen_analysis.groupby('GeneratorType').agg({
                'Generator': 'count',
                'Hours_With_Reserves': 'sum',
                'Total_Hours_Committed': 'sum',
                'Total_Reserves_Provided': 'sum'
            }).reset_index()
            type_summary['Percentage_With_Reserves'] = (type_summary['Hours_With_Reserves'] / type_summary['Total_Hours_Committed'] * 100)
            print(type_summary.to_string())
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()