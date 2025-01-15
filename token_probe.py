import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

# Probing 함수 정의
def outlier_probing(x, block_num, layer, epoch, iteration):

    x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()

    # 첫 100개의 샘플만 사용
    x = x[:256, :, :]
    x = np.abs(x)
    

    # Max 및 Median 값 계산 (Row-wise)
    max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
    median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]

    # Max를 Median으로 나눈 몫 계산
    epsilon = 1e-8
    safe_median_values = np.where(median_values == 0, epsilon, median_values)
    ratio_values = max_values / safe_median_values

    # 196개의 Min/Median ratio를 각 sample 데이터별로 내림차순 정렬
    ratio_values = np.abs(ratio_values)
    sorted_ratios_per_sample = np.sort(ratio_values, axis=1)[:, ::-1]  # Shape: [10, sequence_length]

    # CSV 파일 저장
    save_dir = f"/home/shkim/QT_DeiT_small/reproduce/token_probing_results/{layer}"
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_min_median_ratios.csv")

    # 정렬된 결과를 DataFrame으로 저장
    columns = [f"Col{i+1}" for i in range(197)]
    df = pd.DataFrame(sorted_ratios_per_sample, columns=columns)
    df.to_csv(csv_file_path, index=False)


# def outlier_probing_not_sorted(x, block_num, layer, epoch, iteration):
#     # 확인: Input tensor shape [BS, sequence_length, channel_dim]
#     sequence_len = x.shape[1]

#     x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()

#     x = x[:100, :, :]
#     x = np.abs(x)
    
#     # Max 및 Median 값 계산 (Row-wise)
#     max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
#     median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]

#     # Max를 Median으로 나눈 몫 계산
#     epsilon = 1e-8
#     safe_median_values = np.where(median_values == 0, epsilon, median_values)
#     ratio_values = max_values / safe_median_values


#     ratio_values = np.abs(ratio_values)
#     # ratio_values를 정수형으로 변환
#     ratio_values = ratio_values.astype(int)


#     # CSV 파일 저장
#     save_dir = f"/home/shkim/QT_DeiT_small/reproduce/token_probing_results_not_sorted/{layer}"
#     os.makedirs(save_dir, exist_ok=True)
#     csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_min_median_ratios.csv")

#     # 정렬된 결과를 DataFrame으로 저장
#     columns = [f"Col{i+1}" for i in range(sequence_len)]
#     df = pd.DataFrame(ratio_values, columns=columns)
#     df.to_csv(csv_file_path, index=False)


def norm_probing_not_sorted(x, block_num, layer, epoch, iteration):
    sequence_len = x.shape[1]

    # GPU에서 실행되는 경우 numpy로 변환
    x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()

    # 첫 100개의 샘플만 사용
    x = x[:256, :, :]
    x = np.abs(x)
    
    # Max 및 Median 값 계산 (Row-wise)
    max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
    median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]
    
    # 저장 디렉토리 생성
    save_dir = f"/home/shkim/QT_DeiT_small/reproduce/token_probing_results_not_sorted/{layer}"
    os.makedirs(save_dir, exist_ok=True)
    
    # CSV 파일 경로 설정
    max_csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_max.csv")
    median_csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_median.csv")

    # max_values를 DataFrame으로 저장 (정수형)
    max_values_df = pd.DataFrame(max_values.astype(int), columns=[f"Col{i+1}" for i in range(sequence_len)])
    max_values_df.to_csv(max_csv_file_path, index=False)
    
    # median_values를 DataFrame으로 저장 (소수점 4자리)
    median_values_df = pd.DataFrame(np.round(median_values, 4), columns=[f"Col{i+1}" for i in range(sequence_len)])
    median_values_df.to_csv(median_csv_file_path, index=False)
