import matplotlib
matplotlib.use('Agg')

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

def get_shortest_path_and_length(start_label, end_label, graph):
    """
    두 노드 레이블 사이의 최단 경로(노드 리스트)와 거리를 반환
    """
    try:
        distance, path = nx.single_source_dijkstra(graph, source=start_label, target=end_label, weight='weight')
        return path, distance
    except nx.NodeNotFound as e:
        print(f"  [경로 탐색 오류] 노드 '{e}'가 그래프에 없습니다.")
        return None, 0.0
    except nx.NetworkXNoPath:
        print(f"  [경로 탐색 오류] '{start_label}'와(과) '{end_label}' 사이에 경로가 없습니다.")
        return None, 0.0
    except Exception as e:
        print(f"  [경로 탐색 오류] 예상치 못한 오류: {e}")
        return None, 0.0

def plot_ai_solution(
    ai_path_labels, 
    loaded_graph_data: dict, 
    background_image_data,
    save_filename='app/images/ai_route_map.png'
):
    # --- 1. .pkl 파일 ---
    try:
        graph = loaded_graph_data['g_base']
        node_positions = loaded_graph_data['pos'] 
        print(f"[INFO] app.state에서 G_base와 좌표 정보를 로드했습니다.")
    except KeyError as e:
        print(f"[오류] .pkl 데이터 객체에서 키를 찾을 수 없음: {e}")
        return
    except Exception as e:
        print(f"[오류] .pkl 데이터 객체 처리 중 오류: {e}")
        return

    # ---  2. 배경 이미지 로드  ---
    if background_image_data is None:
        print(f"[오류] 배경 이미지 데이터가 'None'입니다. (서버 시작 시 로드 실패 확인)")
        return
    
    #  이미 로드된 객체를 'img' 변수에 할당
    img = background_image_data
    print(f"[INFO] app.state에서 background_image를 로드했습니다.")

    # --- 3. 이미지 비율에 맞춰 Figure 크기 동적 설정 ---
    img_height, img_width, _ = img.shape
    base_size = 15.0
    if img_height >= img_width:
        figsize = (base_size * (img_width / img_height), base_size)
    else:
        figsize = (base_size, base_size * (img_height / img_width))

    
    plt.figure(figsize=figsize) 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    
    viz_pos = node_positions 
    plt.imshow(img, aspect='equal') 

    # 경로 계산
    total_distance = 0.0; all_path_nodes = set(); all_path_edges = []
    for i in range(len(ai_path_labels) - 1):
        label1, label2 = ai_path_labels[i], ai_path_labels[i+1]
        sub_path, sub_distance = get_shortest_path_and_length(label1, label2, graph)
        if sub_path:
            total_distance += sub_distance; all_path_nodes.update(sub_path); all_path_edges.extend(list(zip(sub_path, sub_path[1:])))
    
    # 플롯
    nx.draw_networkx_edges(graph, viz_pos, edgelist=all_path_edges, edge_color='red', width=2.0)
    start_node, end_node = ai_path_labels[0], ai_path_labels[-1]
    waypoint_nodes = set(ai_path_labels[1:-1]) 
    nodelist = list(all_path_nodes); node_colors = []; node_sizes = []
    for node in nodelist:
        if node == start_node: (node_colors.append('lime'), node_sizes.append(300))
        elif node == end_node: (node_colors.append('magenta'), node_sizes.append(300))
        elif node in waypoint_nodes: (node_colors.append('orange'), node_sizes.append(200))
        else: (node_colors.append('red'), node_sizes.append(0))
    nx.draw_networkx_nodes(graph, viz_pos, nodelist=nodelist, node_color=node_colors, node_size=node_sizes)
    labels_dict = {node: str(idx) for idx, node in enumerate(ai_path_labels[1:-1], 1)}
    labels_dict[start_node] = 'S'; labels_dict[end_node] = 'E'
    nx.draw_networkx_labels(graph, viz_pos, labels=labels_dict, font_color='black', font_weight='bold', font_size=12)
    
    plt.axis('off') # 축 끄기

    if save_filename:
        plt.savefig(
            save_filename, 
            bbox_inches='tight',  
            pad_inches=0,
        )
    print(f"[INFO] 이미지 저장 완료.")
    plt.close()