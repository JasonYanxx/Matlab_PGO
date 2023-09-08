## 文件组织

### Data

- location: N06_TAES\Matlab_PGO

| File name                     | Des.                      | Note                   |
| ----------------------------- | ------------------------- | ---------------------- |
| **mnav_zmp1_jan**             | CORS: 1-31 Jan, 2022      |                        |
| **Urban_dd_0816**             | Urban: 18 min             | z-compensation         |
| **Least_square_dd_urbandata** | Urban: DGNSS localization | Sat: 30~35 deg, SNR>40 |
| urbandd_pe                    | Urban: SPP localization   | Sat: 30~35 deg, SNR>40 |
| dd_result_all_sat             | Urban: DGNSS localization | use all satellites     |
| GT.kml                        | Urban: receiver location  |                        |

### Code

#### Main codes

- location: N06_TAES\Matlab_PGO

| File name            | Des.                                                    | Note |
| -------------------- | ------------------------------------------------------- | ---- |
| **Yan_paper.m**      | plot all figures for paper                              | draw |
| **Yan_sum_exp.m**    | show the usage of main functions                        |      |
| **Yan_exp_record.m** | a chronological record of all experiments and ideas     |      |
| **Yan_functions.m**  | A self-created lib which includes all usefull functions | lib  |

#### Other codes

- location: N06_TAES\Matlab_PGO

| File name                 | Des.                                                                               | Note        |
| ------------------------- | ---------------------------------------------------------------------------------- | ----------- |
| **Yan_explore_stable.m**  | explore  the ability of stable distribution for bounding heavy-tailed distribution |             |
| **stable_overbound.m**    | the realization of stable overbound                                                | function    |
| find_gama.m               | find the optimal gama for stable overbound                                         | function    |
| find_alpha.m              | find the optimal alpha for stable overbound                                        | function    |
| test_gaussian_overbound.m | Integrate T1_trans method in Blanch's two-step Gaussian overbound lib              | From Blanch |
| plot_cdfs.m               | Integrate T1_trans method in Blanch's two-step Gaussian overbound lib              | From Blanch |

### Document

| File                         | Des.                                      | Location            |
| ---------------------------- | ----------------------------------------- | ------------------- |
| Two scenarios for DGNSS.docx | Task file for yihan to process data       | N06_TAES            |
| readme.md                    | code and data orgnization for PGO project | N06_TAES\Matlab_PGO |

## 代码逻辑

### (1) 数据预处理

#### 数据类型

| Type  | Des.                                                   | Time span      |
| ----- | ------------------------------------------------------ | -------------- |
| CORS  | two reference stations (MNAV and ZMP1) in  Minneapolis | Jan 1-31, 2020 |
| Urban | U-blox Zed F9P, near the sea, Hong Kong                | 18 minutes     |

#### DGNSS error calculation

- Form **yihan**
- Data location: N06_TAES\Matlab_PGO\mnav_zmp1_jan
- Data location: N06_TAES\Matlab_PGO\Urban_dd_0816

#### DGNSS positioning

- Use least square (LS), form **yihan**
- Data location: N06_TAES\Matlab_PGO\Least_square_dd_urbandata

#### SPP positioning (option)

- DGNSS error is used to  **compensate** single pseudorange measurement 
- Use weighted least square (WLS), form **yihan**
- Data location: N06_TAES\Matlab_PGO\urbandd_pe

### (2) 数据选择

#### CORS data

- 根据 Larson 的文章，我们直接选择了 $\text{Ele}\in (30,35)$ 的数据用于 DGNSSS error modeling
- 绘制其分布，观察到了 heavy tails

#### Urban data

- 绘制 DNGSS error against SNR and elevation angle
- 观察到 $\text{SNR}>40$ and  $\text{Ele}>30$ 的数据质量较好，可以用于定位和误差建模
- 选用 $\text{SNR}>40$ and  $\text{Ele}\in (30,35)$ 的数据用于 worst-case error modeling
- 选用 $\text{SNR}>40$ and  $\text{Ele}>30$ 的数据用于 DGNSS positioning

### (3) Overbound

#### Method used in paper

- Demo: N06_TAES\Matlab_PGO\Yan_paper.m

| Method             | Implement detail                                   | Note                                                                                         |
| ------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Two-step Gaussian  | set the bias term to be the same                   | beacause we need to calculate PLs                                                            |
| Gaussian Pareto    | the core part is realized by the two-step Gaussian |                                                                                              |
| Principal Gaussian | use inflate factor to inflate the fitted GMM       | the raw fitted GMM is not conservative enough and can badly bound the empirical distribution |

#### Other methods

- Demo: N06_TAES\Matlab_PGO\Yan_sum_exp.m

| Method                    | Implement detail                                                     | Note                                                       |
| ------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| Paired Principal Gaussian | 按照median对分布进行切分，对各部分进行镜像增广，然后分别对增广的数据进行 Principal Gaussian overbound | convolution stability is not satisified; **to be explore** |
| Cauchy overbound          | Too conservative                                                     | **to be explore**                                          |
| Stable overbound          | Hard to obtain optimal params                                        | **to be explore**                                          |

### (4) PL

#### Convolution

- 分布的卷积的两种实现方式 fft & direct convolution

- 相关函数：Yan_functions@**distConv_org**

- 测试函数是否正确：Yan_exp_record.m: <u>compare fft & conv (20230715)</u>

#### PL

- Data

| Item                            | Location                                                          |
| ------------------------------- | ----------------------------------------------------------------- |
| DGNSS positioning error         | N06_TAES\Matlab_PGO\Data\Least_square_dd_urbandata\error_LSDD.csv |
| S_matrix from DNGSS positioning | N06_TAES\Matlab_PGO\Data\Least_square_dd_urbandata\DD_S_matrix    |
| overbound                       | From overbounding methods                                         |

- 相关函数： Yan_functions@**cal_PL**

- 实现：Yan_paper.m: <u>VPL and VPE series, PL compuation time: Fig. 7a,b</u>

## 论文作图

#### Matlab

- Location: WorkAAE\N06_TAES\Matlab_PGO

| Folder          | Des.                                             |
| --------------- | ------------------------------------------------ |
| figure_paper    | for paper use                                    |
| figure_analysis | analysis problems in data from **yihan**         |
| figures         | explore how converstive overbound can affect FDE |

#### Visio

- location: N06_TAES\figure\visio_PGO.vsdx

#### Adobe illustrator

- to do...
