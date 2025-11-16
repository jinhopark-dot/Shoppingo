import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SAVE_DIR = os.path.join(SCRIPT_DIR, 'results')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from app.ai.get_soluton import load_ai_assets, run_ai_inference
    from app.ai.mip_solver import solve_gurobi
except ImportError:
    exit()
    
def create_comparison_plots(file_list, save_dir):
    """
    Pickle 파일 리스트에서 데이터를 로드하여
    1. Bar Chart for Optimality Gap 
    2. Grouped Bar Chart for Time 
    를 생성하고 이미지 파일로 저장합니다.
    """
    
    # 1. 데이터 로드
    gap_plot_data = []
    time_plot_data_long = []

    # 2. Pickle 파일들을 순회하며 데이터 로드 및 가공
    for pkl_file in file_list:
        try:
            size = pkl_file.split('_')[-1].split('.')[0]
            problem_size_label = f'N={int(size)}'
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            cost_key = f'cost_{size}'
            time_key = f'time_{size}'
            
            if cost_key not in data or time_key not in data:
                print(f"  [경고] '{pkl_file}'에 키가 없습니다. 건너뜁니다.")
                continue

            cost_list = data[cost_key]
            time_list = data[time_key]

            # 3. 데이터 가공
            for i in range(len(cost_list)):
                ai_cost, gp_cost = cost_list[i]
                ai_time, gp_time = time_list[i]

                if ai_cost == -1 or gp_cost == -1 or ai_time == -1 or gp_time == -1:
                    continue

                gap = 0.0
                if gp_cost > 0:
                    gap = ((ai_cost - gp_cost) / gp_cost) * 100.0
                gap_plot_data.append({
                    'Problem Size': problem_size_label,
                    'Optimality Gap (%)': gap
                })
                
                time_plot_data_long.append({
                    'Problem Size': problem_size_label, 'Solver': 'AI', 'Time (s)': ai_time
                })
                time_plot_data_long.append({
                    'Problem Size': problem_size_label, 'Solver': 'Gurobi', 'Time (s)': gp_time
                })
                
        except Exception as e:
            print(f"  [오류] '{pkl_file}' 처리 중 오류 발생: {e}")

    # --- 4. Plot 1: 솔루션 품질 (Average Optimality Gap) ---
    if not gap_plot_data:
        print("  [오류] Plot 1 (Cost)을 그릴 유효한 데이터가 없습니다.")
    else:
        df_gap = pd.DataFrame(gap_plot_data)
        size_order = sorted(df_gap['Problem Size'].unique(), key=lambda x: int(x.split('=')[1]))

        print("Plot 1 (Optimality Gap) 생성 중...")
        plt.figure(figsize=(12, 7)) 
        
        ax = sns.barplot(
            x='Problem Size',
            y='Optimality Gap (%)',
            data=df_gap,
            order=size_order,
            palette='viridis',
            errorbar=None 
        )
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                x = p.get_x() + p.get_width() / 2.,
                y = height + 0.025,
                s = f'{height:.3f}%',
                ha = 'center',
                fontsize=24
            )
        
        current_ylim = plt.ylim()
        plt.ylim(current_ylim[0] - 0.025, current_ylim[1] * 1.15)
        
        plt.axhline(y=0, color='#ff595e', linestyle='-', linewidth=5, label='Gurobi (Optimal)')
        plt.title('Average Solution Quality (vs. Exact Method)', fontsize=22, pad=20)
        plt.xlabel('Problem Size (N = S + E + T nodes)', fontsize=18)
        plt.ylabel('Average Optimality Gap (%)', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout()
        
        output_filename_cost = os.path.join(save_dir, 'plot_1_optimality_gap.png')
        plt.savefig(output_filename_cost, dpi=300) 
        print(f"Plot 1이 '{output_filename_cost}'으로 저장되었습니다.")
        plt.close()

    # --- 5. Plot 2: 솔루션 속도 (Average Time) ---
    if not time_plot_data_long:
        print("  [오류] Plot 2 (Time)를 그릴 유효한 데이터가 없습니다.")
    else:
        df_time = pd.DataFrame(time_plot_data_long)
        size_order = sorted(df_time['Problem Size'].unique(), key=lambda x: int(x.split('=')[1]))
        solver_order = ['AI', 'Gurobi']

        print("Plot 2 (Time) 생성 중...")
        plt.figure(figsize=(12, 7))
        
        ax = sns.barplot(
            x='Problem Size',
            y='Time (s)',
            hue='Solver',
            data=df_time,
            order=size_order,
            hue_order=solver_order,
            palette=['#1982c4', '#ff595e'],
            errorbar=None 
        )
        
        ax.set_yscale('log')
        
        for p in ax.patches:
            height = p.get_height()
            if height <= 0:
                continue
                
            ax.text(
                x = p.get_x() + p.get_width() / 2.,
                y = height * 1.11, 
                s = f'{height:.2f}s',
                ha = 'center',
                fontsize=20
            )

        current_ylim = plt.ylim()
        plt.ylim(current_ylim[0], current_ylim[1] * 1.2) 

        plt.title('Average Solution Speed (vs. Exact Method)', fontsize=22, pad=20)
        plt.xlabel('Problem Size (N = S1 + E1 + T nodes)', fontsize=18)
        plt.ylabel('Average Time (seconds) \n [Log Scale]', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(title='Solver', fontsize=16, title_fontsize=17)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout()
        
        output_filename_time = os.path.join(save_dir, 'plot_2_time.png')
        plt.savefig(output_filename_time, dpi=300)
        print(f"Plot 2가 '{output_filename_time}'으로 저장되었습니다.")
        plt.close()


def compare_model(instance_size, instance_num):
    best_cost = []
    time_list = []
    for i in range(instance_num):
        print(f'  ---------- Iteration {i+1} ----------')
        instance = [f'T{i}' for i in np.random.choice(range(1, 329), instance_size-2, replace=False)] + ['S1', 'E1']
        ai_start = time.time()
        final_path_labels_ai, best_cost_ai = run_ai_inference(ai_model, opts, full_data, instance, start_node_label = 'S1', num_samples=1000)
        ai_time = time.time() - ai_start
        
        gp_start = time.time()
        final_path_labels_gp, best_cost_gp = solve_gurobi(full_data, instance, 'S1', opts)
        gp_time = time.time() - gp_start
        
        time_list.append([np.round(ai_time, 4), np.round(gp_time, 4)])
        best_cost.append([best_cost_ai, best_cost_gp])
    
    return best_cost, time_list

if __name__ == "__main__":
    load_path = os.path.join(PROJECT_ROOT, 'app/ai/outputs/shopping_30/shopping_run_20251017T155424/best_model.pt')
    full_graph_path = os.path.join(PROJECT_ROOT, 'app/ai/full_shortest_paths.pkl')
    ai_model, opts, full_data = load_ai_assets(load_path, full_graph_path)

    # 비교할 인스턴스 크기 목록
    instance_sizes = [12, 14, 16, 18] 
    n = 30 

    file_paths_for_plot = []
    for size in instance_sizes:
        print(f'---------- Size {size} ----------')
        cost_list, time_list = compare_model(instance_size=size, 
                                            instance_num=n)

        result_data = {
            f'cost_{size}': cost_list, 
            f'time_{size}': time_list
        }
        
        filename = os.path.join(SAVE_DIR, f'experiment_results_{size}_a.pkl')
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(result_data, f)
            print(f"결과가 {filename}에 성공적으로 저장되었습니다.")
            file_paths_for_plot.append(filename)
        except Exception as e:
            print(f"파일 저장 중 오류 발생 ({filename}): {e}")
            
    if file_paths_for_plot:
        create_comparison_plots(file_paths_for_plot, SAVE_DIR)