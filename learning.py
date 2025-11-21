import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


def train_diabetes_model():
    print(">>> [1/6] 데이터 로딩 및 변수명 통일")

    # 로드하자마자 대문자로 변환
    # 21년도자료의 칼럼명이 소문자로 되어있기때문
    def load_data(path, year_label):
        try:
            d = pd.read_sas(path)
            d.columns = d.columns.str.upper()
            print(f"- {year_label} 로드: {len(d)}명")
            return d
        except:
            print(f"⚠️ 파일 없음: {path}")
            return None

    # 데이터 로드
    df_23 = load_data('hn23_all.sas7bdat', '2023년')
    df_22 = load_data('hn22_all.sas7bdat', '2022년')
    df_21 = load_data('hn21_all.sas7bdat', '2021년')

    # 병합
    df_list = [d for d in [df_23, df_22, df_21] if d is not None]
    if not df_list: return

    df = pd.concat(df_list, ignore_index=True)
    print(f"총 통합 데이터: {len(df)}명")

    # 3. 컬럼 선택
    target_cols = [
        'AGE', 'SEX',
        'N_EN', 'N_CHO', 'N_FAT', 'N_PROT', 'N_NA', 'N_SUGAR',
        'HE_GLU', 'DE1_DG', 'DE1_31', 'HE_FST'
    ]

    # 누락 컬럼 확인
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        print(f"필수 컬럼 누락: {missing}")
        return

    df_clean = df[target_cols].copy()

    print(">>> [2/6] 전처리 진행")
    # 결측치 제거 전후 비교 (데이터 손실 확인용)
    before_len = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"- 결측치 제거: {before_len}명 -> {len(df_clean)}명")

    df_clean[['DE1_DG', 'DE1_31']] = df_clean[['DE1_DG', 'DE1_31']].replace([8, 9], 0)

    # 필터링
    df_clean = df_clean[
        (df_clean['AGE'] >= 19) &
        (df_clean['N_EN'] > 500) &
        (df_clean['HE_FST'] >= 8)
        ]

    # 관리 환자 제거
    mask_well_managed = ((df_clean['DE1_DG'] == 1) | (df_clean['DE1_31'] == 1)) & (df_clean['HE_GLU'] < 126)
    df_final = df_clean[~mask_well_managed].copy()

    print(f"최종 학습 데이터: {len(df_final)}명 (관리환자 {mask_well_managed.sum()}명 제외됨)")

    # 데이터 분리
    y = (df_final['HE_GLU'] >= 126).astype(int)
    X = df_final[['AGE', 'SEX', 'N_EN', 'N_CHO', 'N_FAT', 'N_PROT', 'N_NA', 'N_SUGAR']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE 적용 (당뇨 환자의 수 가 적어서 인위적으로 증강)
    print(">>> [4/6] SMOTE 데이터 증강")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"- 위험군 수: {sum(y_train)}명 -> {sum(y_train_res)}명 (증강됨)")

    # 모델 학습
    print(">>> [5/6] 모델 학습 시작")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    # max_depth를 제한하면 과적합을 막아 일반화 성능(Recall)이 조금 오를 수 있다??
    model.fit(X_train_res, y_train_res)

    # 결과 확인
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 확률값

    print("\n" + "=" * 30)
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("=" * 30)

    # 중요도 출력
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(5))

    joblib.dump(model, 'diabetes_risk_model_final.pkl')


if __name__ == "__main__":
    train_diabetes_model()