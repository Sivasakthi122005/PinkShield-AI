"""Performance page: every number is read from the saved metrics files."""
import json
import os

MODELS = 'models'


def _load(name):
    path = os.path.join(MODELS, name)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def headline_stats():
    """Hero-card numbers (strings) - '-' when a metrics file is missing."""
    hist, tab, fl = _load('training_history.json'), _load('tabular_metrics.json'), _load('fl_metrics.json')
    img = f"{max(hist['val_acc']) * 100:.2f}" if hist and hist.get('val_acc') else '-'
    tabv = f"{tab['accuracy']:.2f}" if tab else '-'
    fed = f"{fl['global_accuracy'][-1] * 100:.2f}" if fl and fl.get('global_accuracy') else '-'
    return {'image': img, 'tabular': tabv, 'federated': fed}


def _layout(fig, title, h=360, ytitle=None, xtitle=None):
    fig.update_layout(title=title, height=h, template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=40, r=20, t=50, b=40),
                      yaxis_title=ytitle, xaxis_title=xtitle)
    return fig


def render_performance():
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center;">Model Performance</div>', unsafe_allow_html=True)
    st.caption("All values are loaded from the saved metrics files in models/. "
               "Run `python run_all.py` to refresh them.")

    # ---- Module 1: image ------------------------------------------------
    st.subheader("Module 1 - Histopathology image model")
    hist = _load('training_history.json')
    if hist:
        ep = list(range(1, len(hist['train_acc']) + 1))
        fig = go.Figure()
        fig.add_scatter(x=ep, y=[a * 100 for a in hist['train_acc']], name='Train', mode='lines+markers')
        fig.add_scatter(x=ep, y=[a * 100 for a in hist['val_acc']], name='Validation', mode='lines+markers')
        st.plotly_chart(_layout(fig, 'EfficientNet-B0 accuracy per epoch', ytitle='Accuracy (%)', xtitle='Epoch'),
                        use_container_width=True)
        img_m = _load('image_metrics.json')
        if img_m:
            c = st.columns(4)
            c[0].metric('Val accuracy', f"{img_m['accuracy']}%")
            c[1].metric('Sensitivity', f"{img_m['sensitivity']}%")
            c[2].metric('Specificity', f"{img_m['specificity']}%")
            c[3].metric('AUC', img_m['auc_roc'])
            st.caption(f"Split: {img_m.get('split', 'n/a')}")
        else:
            st.info("Image metrics (sensitivity / specificity / AUC) appear after re-training with "
                    "`python train_image_model.py` (patient-level split).")
    else:
        st.info("No image training history found.")

    # ---- Module 2: tabular ------------------------------------------------
    st.subheader("Module 2 - Clinical (Wisconsin) model")
    tab, ev = _load('tabular_metrics.json'), _load('evaluation_report.json')
    if tab:
        c = st.columns(4)
        c[0].metric('Test accuracy', f"{tab['accuracy']}%")
        c[1].metric('Sensitivity', f"{tab['sensitivity']}%" if 'sensitivity' in tab else 'n/a')
        c[2].metric('Specificity', f"{tab['specificity']}%" if 'specificity' in tab else 'n/a')
        c[3].metric('AUC-ROC', tab['auc_roc'])
    if ev and ev.get('tabular_cv'):
        cv = ev['tabular_cv']
        st.markdown(f"**{cv['folds']}-fold cross-validation** ({ev.get('tabular_model', 'model')}): "
                    f"accuracy {cv['accuracy_mean']}% ± {cv['accuracy_std']}, "
                    f"sensitivity {cv['sensitivity_mean']}% ± {cv['sensitivity_std']}, "
                    f"AUC {cv['auc_mean']} ± {cv['auc_std']}")
        if ev.get('roc'):
            fig = go.Figure()
            fig.add_scatter(x=ev['roc']['fpr'], y=ev['roc']['tpr'], name='Clinical model', mode='lines')
            fig.add_scatter(x=[0, 1], y=[0, 1], name='Chance', mode='lines', line=dict(dash='dot'))
            st.plotly_chart(_layout(fig, 'ROC curve (held-out test set)', 320,
                                    'True positive rate', 'False positive rate'),
                            use_container_width=True)
        cm = ev.get('confusion_matrix')
        if cm:
            fig = go.Figure(go.Heatmap(z=cm, x=['Pred benign', 'Pred malignant'],
                                       y=['True benign', 'True malignant'],
                                       text=cm, texttemplate='%{text}', colorscale='Blues'))
            st.plotly_chart(_layout(fig, 'Confusion matrix (test set)', 300), use_container_width=True)
    elif not tab:
        st.info("Run `python run_all.py` to train and evaluate the clinical model.")

    # ---- Module 3: NLP ----------------------------------------------------
    st.subheader("Module 3 - Symptom text model")
    nlp = _load('nlp_metrics.json')
    if nlp:
        c = st.columns(2)
        c[0].metric('Accuracy (held-out templates)', f"{nlp['accuracy']}%")
        c[1].metric('AUC-ROC', nlp['auc_roc'])
        st.warning("Trained on synthetic, template-generated sentences. A saturated score reflects an easy, "
                   "keyword-separable dataset - it is NOT evidence of clinical validity. "
                   "That is why this module carries the lowest fusion weight (15%).")

    # ---- Module 5: federated ----------------------------------------------
    st.subheader("Module 5 - Federated learning (FedAvg, non-IID hospitals)")
    fl = _load('fl_metrics.json')
    if fl:
        cfg = fl.get('config', {})
        fig = go.Figure()
        fig.add_scatter(x=fl['rounds'], y=[a * 100 for a in fl['global_accuracy']],
                        name='Global model (shared test set)', mode='lines+markers', line=dict(width=3))
        for h in 'ABC':
            key = f'hospital_{h}_acc'
            if key in fl:
                fig.add_scatter(x=fl['rounds'], y=[a * 100 for a in fl[key]],
                                name=f'Hospital {h} (local validation)', mode='lines', line=dict(dash='dot'))
        st.plotly_chart(_layout(fig, 'Accuracy per communication round', 380, 'Accuracy (%)', 'Round'),
                        use_container_width=True)
        if fl.get('global_loss'):
            fig = go.Figure()
            fig.add_scatter(x=fl['rounds'], y=fl['global_loss'], name='Global loss', mode='lines+markers')
            st.plotly_chart(_layout(fig, 'Global log-loss per round (should fall)', 280, 'Log-loss', 'Round'),
                            use_container_width=True)
        b = fl.get('baselines')
        if b:
            c = st.columns(3)
            c[0].metric('Federated', f"{b['federated_accuracy'] * 100:.2f}%")
            c[1].metric('Centralized (no privacy)', f"{b['centralized_accuracy'] * 100:.2f}%")
            lo = b['local_only_accuracy']
            c[2].metric('Best local-only', f"{max(lo.values()) * 100:.2f}%")
            st.caption("FedAvg matches centralized training without pooling patient records. "
                       "The Wisconsin data is easy, so local-only models also score well - the privacy "
                       "benefit, not an accuracy gain, is the point of this experiment.")
        if cfg.get('hospital_sizes'):
            st.caption(f"Hospitals (train size / malignant share): " + ' | '.join(
                f"{k}: {cfg['hospital_sizes'][k]} / {cfg['hospital_malignant_share'][k]:.0%}"
                for k in cfg['hospital_sizes']))
    else:
        st.info("Run `python module5_federated/run_simulation.py` to generate federated metrics.")

    st.markdown('</div>', unsafe_allow_html=True)
