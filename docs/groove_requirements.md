# GrooveSim 要件定義メモ

## 目的

入力された音楽音響または音系列から、`grooveness` を

- 身体を周期的に動かしたくなる強さ
- ビートへの同調しやすさ
- 予測可能性と驚きのバランス

として定量化する研究用プログラムを作る。

この文書は、実装前に既存研究を整理し、必要な計算要素と MVP の範囲を定義するためのもの。

## 研究整理

### 1. Groove の中心概念

近年のレビューでは、groove は概ね「音楽に合わせて気持ちよく身体を動かしたくなる傾向」と整理されている。心理・神経科学の文脈では、リズム知覚、運動予測、報酬系の相互作用として扱われることが多い。

- Etani et al. (2024) は、groove を beat / rhythm / body movement / pleasure / reward / predictive coding と結びつけて総説している。
- Senn et al. (2022) は groove の成立条件を「predictability と surprise の sweet spot」と整理している。

### 2. Groove を上げる主因として一貫して報告されるもの

研究を横断すると、以下の因子は比較的一貫して重要:

- `beat salience / pulse clarity`
- `syncopation` や中程度の `rhythmic complexity`
- `event density`
- 低域のリズム的変動、特に `low-frequency spectral flux`
- 聴き手が同調しやすいテンポ帯

一方で、`microtiming` は条件依存で、常に効くわけではない。

### 3. 計算モデル上の含意

groove 推定は単一スカラーではなく、少なくとも次の 4 系統の情報を持つ必要がある。

1. `Auditory drive`
   音響信号そのものがどれだけ拍を強調しているか
2. `Beat entrainment`
   どれだけ安定した周期を誘導するか
3. `Predictive tension`
   予測しやすさと逸脱のバランス
4. `Embodied support`
   身体運動しやすいテンポ帯・低域・周期安定性を持つか

## 必要な計算要素

### A. Auditory front-end

入力音声を、そのまま STFT だけで扱うより、周波数チャネルごとの時間変化として扱う方が良い。

必要要素:

- モノラル化
- リサンプリング
- 短時間フレーム化
- 聴覚フィルタバンク
  - 第一候補: `gammatone-like` または `mel` サブバンド
- チャネルごとの包絡抽出
- 半波整流または差分による `onset / novelty / salience` 強調

研究的には excitation pattern や auditory salience を使う方向が妥当だが、MVP では次を採用する:

- `multi-band spectral flux`
- `sub-band onset envelope`
- `low-band / mid-band / full-band novelty`

理由:

- 聴覚モデルに近い分解表現を保持できる
- librosa/scipy で再現しやすい
- spectral flux は beat 同期・神経同調研究でも有効

### B. Beat entrainment / periodicity

groove には「拍が見えること」が必要だが、完全な単調さだけでは弱い。よって、拍の明瞭さと周期安定性を計算する。

必要要素:

- `tempogram` または周期自己相関
- 拍候補周波数のピーク探索
- 拍の強さ
- 拍の安定性
- 倍テン・半テンの曖昧性

MVP の指標:

- `beat_strength`
  - novelty/tempogram 上の主ピーク強度
- `beat_clarity`
  - 主ピークと周辺ピークのコントラスト
- `tempo_alignment_score`
  - 主テンポが 1-2.5 Hz 程度の身体同調しやすい帯域にあるか
- `periodicity_stability`
  - 時間窓ごとのテンポ推定の分散が小さいか

### C. Syncopation / rhythmic complexity

groove 研究では、中程度の syncopation や complexity が高い groove と結びつくことが多い。単純すぎても複雑すぎても弱くなりやすい。

必要要素:

- onset 列またはビート量子化列
- 仮定 meter 上での強拍/弱拍重み
- 強い位置の休符と弱い位置の発音の比較
- event density

MVP の指標:

- `syncopation_index`
  - 16 分または 8 分格子上での metrical incongruity
- `event_density`
  - 単位時間あたり onset 数
- `complexity_balance_score`
  - 単純すぎる/複雑すぎる両端を下げる逆 U 字型変換

注意:

- 拍と小節の推定が外れると syncopation は崩れる
- したがって meter 推定の信頼度も保持する

### D. Surprisal / prediction error

ユーザーが望んでいる「surprisal を使って身体運動の生起傾向を見る」という方向性は、予測符号化や rhythmic incongruity の研究と整合的。

ただし、音声から直接「認知的 surprisal」を精密に出すのは難しい。MVP では 2 段階に分ける。

#### D1. MVP surprisal

入力の短期文脈から、次時刻に onset が来る確率を推定し、自己情報量で驚きを測る。

- 入力: onset envelope または量子化 onset 列
- 方法:
  - n-gram / Markov 的遷移
  - 周期テンプレートとの残差
  - 自己回帰予測誤差
- 出力:
  - `mean_surprisal`
  - `surprisal_variance`
  - `moderate_surprisal_ratio`

groove に効きやすいのは「驚きが大きいこと」より、
`適度な surprisal が安定拍の上に乗ること` と仮定する。

#### D2. 拡張 surprisal

将来的には以下を追加可能:

- IDyOM 系の情報量計算
- symbolic rhythm model
- audio-to-symbolic 後の階層的予測モデル
- style-conditioned expectation model

### E. Low-frequency drive

自然音楽での groove には低域が強く効く報告が多い。ドラムとベースが拍感と運動衝動を支えるため、低域サブバンドは独立に扱うべき。

必要要素:

- 20-150 Hz 付近のエネルギー変動
- low-band spectral flux
- キック/ベースの周期性

MVP の指標:

- `low_freq_flux`
- `bass_pulse_strength`
- `low_mid_balance`

### F. Listener-independent と listener-dependent の分離

研究上、familiarity や style preference は groove にかなり効くが、まずは信号処理ベースの `listener-independent groove potential` を作るのが妥当。

将来拡張:

- ジャンル条件付きモデル
- 学習済み groove rating 回帰器
- 個人差パラメータ

## 推奨パイプライン

### 入力 1: 音声

1. 音声読み込み
2. モノラル化・正規化・リサンプル
3. 聴覚風サブバンド表現作成
4. 各帯域 novelty / flux / onset envelope 算出
5. 周期性解析
6. 拍候補と meter 候補推定
7. syncopation / density / surprisal 算出
8. 低域寄与を別途算出
9. 特徴量統合
10. 総合 `groove_score` と内訳を返す

### 入力 2: 音系列

対象:

- onset times
- IOI 列
- binary pulse train
- MIDI ノートオン時刻

手順:

1. 量子化候補グリッド推定
2. 周期性・テンポ帯推定
3. syncopation / density / surprisal を直接計算
4. `symbolic_groove_score` を返す

## 出力要件

プログラムは単一スコアだけでなく、説明可能な分解結果を返すべき。

必須出力:

- `groove_score`
- `beat_strength`
- `beat_clarity`
- `tempo_bpm`
- `tempo_alignment_score`
- `syncopation_index`
- `event_density`
- `mean_surprisal`
- `moderate_surprisal_ratio`
- `low_freq_flux`
- `confidence`

推奨出力:

- フレーム系列の groove-related curves
- top contributing features
- 図:
  - novelty curve
  - tempogram
  - surprisal curve
  - beat grid overlay

## MVP の明確化

最初の実装対象は以下に限定する。

### MVP に入れるもの

- 音声入力と onset 列入力の両対応
- auditory-like subband novelty
- tempogram ベースの beat strength / beat clarity
- 簡易 syncopation 指標
- Markov 的または自己回帰的 surprisal
- low-frequency spectral flux
- 線形またはルールベース統合による `groove_score`

### MVP で入れないもの

- EEG/運動データからの直接学習
- 深層学習 end-to-end groove predictor
- 高度な source separation 前提処理
- 個人差モデル
- 厳密な知覚実験を要するパラメータ最適化

## スコア統合の初期方針

初期段階では学習済み回帰器ではなく、説明可能な hand-crafted 統合を使う。

例:

- `entrainment_component`
  - beat_strength, beat_clarity, periodicity_stability, tempo_alignment
- `tension_component`
  - syncopation_index, mean_surprisal, moderate_surprisal_ratio
- `embodiment_component`
  - low_freq_flux, bass_pulse_strength, event_density

統合は以下の考え方で行う:

- beat が弱すぎる場合は全体を下げる
- surprisal / complexity は逆 U 字型で扱う
- 低域駆動は加点する
- 信頼度が低い場合はスコアを圧縮する

## 機能要件

- WAV/MP3/FLAC を読める
- onset 時刻 CSV/JSON からも計算できる
- CLI で実行できる
- 結果を JSON で保存できる
- 特徴量図を PNG で出せる
- パラメータを設定ファイルで変えられる

## 非機能要件

- 研究用途として再現可能であること
- 各特徴量の定義が明示されていること
- スコアの説明可能性があること
- 将来 ML 化できるよう特徴量を分離しておくこと

## 実装上の意思決定

### 採用

- Python
- `numpy`, `scipy`, `librosa`, `soundfile`
- まずは rule-based / feature-based

### 保留

- `essentia` 導入
- 真の聴覚末梢モデルの導入
- source separation
- 学習ベース統合

## 次の実装タスク

1. プロジェクトの雛形を作る
2. audio front-end を実装する
3. beat / periodicity 特徴を実装する
4. symbolic syncopation / surprisal を実装する
5. groove score 統合器を作る
6. CLI と JSON 出力を作る
7. サンプル音源で挙動確認する

## 参考文献メモ

- Etani, Miura, Kawase, Fujii, Keller, Vuust, Kudo (2024), *A review of psychological and neuroscientific research on musical groove*  
  https://www.sciencedirect.com/science/article/pii/S0149763423004918

- Senn, Kilchenmann, Bechtold, Hoesl (2022), *The sweet spot between predictability and surprise: musical groove in brain, body, and social interactions*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9396343/

- Witek et al. (2014), *Syncopation, body-movement and pleasure in groove music*  
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094446

- Madison, Gouyon, Ullen, Hornstrom (2011), *Modeling the tendency for music to induce movement in humans: first correlations with low-level audio descriptors across music genres*  
  https://pubmed.ncbi.nlm.nih.gov/21728462/

- Stupacher, Hove, Janata (2016), *Audio Features Underlying Perceived Groove and Sensorimotor Synchronization in Music*  
  https://www.researchgate.net/publication/291351443_Audio_Features_Underlying_Perceived_Groove_and_Sensorimotor_Synchronization_in_Music

- Stupacher et al. (2019), *Neural entrainment is associated with subjective groove and complexity for performed but not mechanical musical rhythms*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6647194/

- Rajendran, Harper, Schnupp (2020), *Auditory cortical representation of music favours the perceived beat*  
  https://pubmed.ncbi.nlm.nih.gov/32269783/

- Bauer, Stenner, Ghani, Kösem, Scharinger, Obleser, Henry (2022), *Neural synchronization is strongest to the spectral flux of slow music and depends on familiarity and beat salience*  
  https://elifesciences.org/articles/75515

- Senn et al. (2018), *Groove in drum patterns as a function of both rhythmic properties and listeners' attitudes*  
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199604

- Vuust, Dietz, Witek, Kringelbach (2018) など predictive coding / rhythmic incongruity 系文献
  https://www.frontiersin.org/articles/10.3389/fpsyg.2014.01111/full
