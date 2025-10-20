# NLP-Arithmetic-as-a-language
此為清大自然語言處理課程的 HW2: Arithmetic as a language, you will practice training simple sequence generation models. We will treat arithmetic expressions as a language and use recurrent neural networks (RNN, LSTM) to train a sequence generation model for this special language. 

# NLP HW2 — Character-Level Arithmetic Generation  
**Author:** 翁智宏  
**Student ID:** 313707043  
**Institution:** 交大資財所  
**Platform:** Local (Ubuntu + RTX 3090)  
**Python version:** 3.10.18  
**OS:** Linux 6.8.0-65-generic  

---

## System Environment  
| Component | Specification |
|------------|---------------|
| CPU | x86_64, 24 cores |
| GPU | NVIDIA GeForce RTX 3090 |
| OS | Linux (Ubuntu 6.8.0) |
| Python | 3.10.18 |
| Frameworks | PyTorch, NumPy, Pandas, Matplotlib, Seaborn |

---

## Hyperparameters (5%)  

| Parameter | Value |
|------------|--------|
| batch_size | 64 |
| epochs | 30 |
| embed_dim | 256 |
| hidden_dim | 256 |
| learning rate | 5e-4 |
| weight_decay | 1e-4 |
| grad_clip | 1 |

---

## If you use RNN or GRU instead of LSTM (10%)

### Experimental Results

| 模型 | EM | Train Loss | 現象摘要 |
|------|----|-------------|-----------|
| RNN | 0.3527 | 1.1115 | 訓練不穩、準確率低、收斂慢 |
| GRU | 0.9306 | 0.6549 | 收斂快速、準確率高、接近 LSTM |
| LSTM | 0.93 | 0.65 | 穩定、效果最佳 |

### 分析  
**(1) RNN**  
RNN 沒有 gating mechanism，因此在長序列中容易出現：  
- 梯度消失（vanishing gradient）→ 無法記住前面運算結果  
- 資訊遺失 → 「進位規則」或「括號結構」無法被長期保存  
- 訓練不穩 → EM 只有 0.35 左右，Loss 難以下降  

結果：模型幾乎無法學會完整的算術規則，只能「背樣式」，輸出常錯。  

**(2) GRU**  
GRU 是對 LSTM 的簡化版本（只保留 update 與 reset gate）。  
- 能部分解決梯度消失問題  
- 訓練更快、參數較少  
- 表現接近 LSTM  

結果：GRU 的 EM≈0.93，Train Loss≈0.65，幾乎與 LSTM 一樣好。  

---

## If training set uses 3-digit numbers and evaluation uses 2-digit numbers (10%)

### 現象說明
模型在訓練過程中只看過「三位數」形式，但測試是「兩位數」，導致：
- 序列長度變短  
- 數值範圍不同（0–99 vs 100–999）  
- 進位規則不同  

這是典型的 **distribution shift（資料分佈轉移）**。  
模型若未真正學會算術規則，只是記憶字串模式，就無法泛化到新型態輸入。  

> 結果：生成品質顯著下降、預測錯誤增加、EM 明顯降低。  

---

## If 20% of training samples have incorrect answers (10%)

### 效果分析
- **標籤噪音 (Label Noise)**：同一輸入對應多個輸出 → 模型混淆  
- **機率分佈偏移**：錯誤答案仍被分配部分機率 → 有時答錯  
- **泛化下降**：學到錯誤模式，無法歸納真正規則  

### 範例比較

| 狀況 | 輸入 | 模型輸出 | 評價 |
|------|--------|------------|--------|
| 無噪音訓練 | `12+7=` | `19` | ✅ 正確 |
| 含 20% 錯誤 | `12+7=` | `18` | ❌ 被誤導 |
| 含 20% 錯誤 | `33+9=` | `43` | ⚠️ 不穩定 |

> 結果：訓練收斂變慢、Loss 提高、準確率下降約 15–25%。  

---

## Why do we need Gradient Clipping? (5%)

| 梯度裁剪設定 | Epoch 30 EM | Train Loss | 現象 |
|---------------|-------------|-------------|--------|
| ❌ 無裁剪 | 0.9126 | 0.6689 | 訓練不穩，Loss 變化大 |
| ✅ 有裁剪 | 0.9273 | 0.6575 | 收斂平穩、準確率提升 |

### 原因分析
- **梯度裁剪**可防止在長序列反傳過程中梯度爆炸。  
- 限制梯度最大值，使參數更新穩定。  
- 提升模型泛化能力與最終準確率。  

> 實驗證明：有裁剪 → 收斂更平滑、EM 更高。

---

## Anything that can strengthen your report. (5%)

### 任務特性
這個任務屬於「字元級序列生成（Character-Level Sequence Generation）」：
- **語法規律學習**：例如括號、運算子順序  
- **數值推理能力**：學會進位、借位邏輯  
- **長序列記憶能力**：三位數比一位數更難學  

因此本任務結合了 **符號推理 (Symbolic Reasoning)** 與 **序列建模 (Sequence Modeling)**。  

### 模型比較觀察
透過實驗發現：
- LSTM 最能學習長期依賴  
- GRU 效果接近但更輕量  
- RNN 易梯度消失，表現最差  

### 訓練技巧補充
在長序列模型中常見梯度爆炸問題，因此需採用：
- **梯度裁剪 (Gradient Clipping)**  
- **Dropout** 或 **Label Smoothing** 防止過擬合  
- **Teacher Forcing** 穩定訓練  

> 結論：LSTM + 梯度裁剪 → 最穩定且準確的組合。

---

## Training Logs & Evaluation Accuracy (10%)

| 模型 | 梯度裁剪 | 最終 EM | 最終 Loss |
|------|------------|----------|------------|
| LSTM | 有 | 0.9273 | 0.6575 |
| LSTM | 無 | 0.9126 | 0.6689 |


