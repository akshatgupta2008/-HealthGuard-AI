import sys, traceback
try:
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from pathlib import Path

    p = Path('src/model_pipeline.joblib')
    if not p.exists():
        print('MODEL_NOT_FOUND')
        sys.exit(2)

    pl = joblib.load(p)
    metrics = getattr(pl, 'metrics', None)
    if metrics is None or 'xgboost' not in metrics:
        print('METRICS_MISSING')
        sys.exit(3)

    importances = metrics['xgboost'].get('feature_importances')
    print('Feature importances type:', type(importances))
    try:
        items = list(importances.items())
    except Exception as e:
        print('IMPORTANCES_NOT_DICT', repr(e))
        traceback.print_exc()
        sys.exit(4)

    df_imp = pd.DataFrame(items, columns=['Feature','Importance']).sort_values('Importance', ascending=False)
    print(df_imp.head().to_string())

    plt.figure(figsize=(6,4))
    sns.barplot(data=df_imp.head(10), x='Importance', y='Feature')
    plt.tight_layout()
    out = 'test_xgb_importances.png'
    plt.savefig(out)
    print('SAVED_PLOT', out)
except Exception:
    traceback.print_exc()
    sys.exit(1)
