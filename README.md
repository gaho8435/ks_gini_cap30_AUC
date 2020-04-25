# ks_gini_cap30_AUC

## 簡介
此為計算模型結果參數工具，包含計算ks,gini,cap30,AUC及ROC curve畫圖：

  
* ks_gini_cap30_AUC:
  存放主要程式碼
* data_processing:
  demo資料處理程式碼
* example:
  存放demo檔

## 參數介紹
* ks:
* gini:
* cap30:
* AUC:
  
## Document
class ks_gini_cap30_AUC.ks_gini_cap30_AUC(classes,model_predict_proba,y,do_cate=True)

**Parameters:**
* classes:int,類別個數
* model_predict_proba:array,模型預測機率值
* y:array,真實結果
* do_cate:bool,是否類別化
  
**Methods:**
* calculate_auc(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* calculate_cap30(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* calculate_ks(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* calculate_gini(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* calculate_all(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* calculate_detail(self, num = 1)
  * Parameters: \
  **num:int,類別代號**
* ROC_AUC_plot(self, Title='', figsize = (10,8), fontsize = 12, save = None)
  * Parameters: \
  **Title:str,圖片標題** \
  **figsize:(float,float),圖片長寬大小** \
  **fontsize:float,字體大小**
  **save:str,儲存圖檔名稱(None時不存檔)**

## Demo
例子存放於example中，以下介紹：

#### 輸入套件
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
```

#### 輸入資料
```python
data_x = pd.read_csv('data\data_x.csv')
data_y = pd.read_csv('data\data_y.csv')
```

#### train & test set
```python
train_x , test_x , train_y , test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
```

#### 建立RandomForest Model
```python
clf = RandomForestClassifier(n_estimators=100,
                             n_jobs=-1,
                             random_state=42)
```

#### Model fit
```python
clf.fit(train_x,train_y)
```

#### import 此計算工具
```python
from ks_gini_cap30_AUC import ks_gini_cap30_AUC
```

#### 建立ks_gini_cap30_AUC物件
```python
result = ks_gini_cap30_AUC(classes = 2, model_predict_proba = clf.predict_proba(test_x), 
                           y = np.array(test_y), do_cate = True)
```

#### 
```python
result.ROC_AUC_plot(figsize = (8,6),fontsize = 14,save = 'ROC_curve')
```

![ROC_curve](fig/ROC_curve.png)

#### 
```python
result.calculate_ks(num = 1)

> 0.4896105491168129
```

#### 
```python
result.calculate_gini(num = 1)

> 0.899977110828253
```

#### 
```python
result.calculate_cap30(num = 1)

> 0.782608695652174
```

#### 
```python
result.calculate_auc(num = 1)

> 0.8662566824928035
```

#### 
```python
result.calculate_all(num = 1)
```

|  | ks | gini | cap30 | auc |
|--|----|------|-------|-----|
| 1 | 0.489611 | 0.899977 | 0.782609 | 0.866257 |


#### 
```python
result.calculate_detail(num = 1)
```
|    |   rank |   人數 |   累積人數 |   y |   累積y |        y率 |        KS |        Gini |
|---:|-------:|-------:|-----------:|----:|--------:|-----------:|----------:|------------:|
|  0 |      1 |    142 |        142 |  99 |      99 | 0.697183   | 0.291022  | 0.039241    |
|  1 |      2 |    141 |        283 |  53 |     152 | 0.375887   | 0.400932  | 0.0208599   |
|  2 |      3 |    142 |        425 |  46 |     198 | 0.323944   | 0.482467  | 0.0182332   |
|  3 |      4 |    141 |        566 |  27 |     225 | 0.191489   | 0.489611  | 0.0106267   |
|  4 |      5 |    142 |        708 |   6 |     231 | 0.0422535  | 0.413043  | 0.00237824  |
|  5 |      6 |    142 |        850 |   7 |     238 | 0.0492958  | 0.340429  | 0.00277461  |
|  6 |      7 |    141 |        991 |   9 |     247 | 0.0638298  | 0.276426  | 0.00354224  |
|  7 |      8 |    142 |       1133 |   1 |     248 | 0.00704225 | 0.180096  | 0.000396373 |
|  8 |      9 |    141 |       1274 |   4 |     252 | 0.0283688  | 0.0963299 | 0.00157433  |
|  9 |     10 |    142 |       1416 |   1 |     253 | 0.00704225 | 0         | 0.000396373 |

## TODO

