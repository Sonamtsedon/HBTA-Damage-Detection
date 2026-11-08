"""HBTA Dense Autoencoder baseline (proposed; confirm with supervisor/team).
Usage: python hbta_baseline_autoencoder.py --data /path/to/data_100Hz.h5
Requires: pip install numpy pandas h5py scikit-learn tensorflow matplotlib joblib
"""
import argparse
import json
from pathlib import Path
import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
WINDOW = 500
HEALTHY = 'MVS_P1_UDS_SM_Y_01'
REFERENCE = 'MVS_P1_UDS_SM_Y_02'
DAMAGE = {f'DS{i}': f'MVS_P1_DS{i}_SM_Y_01' for i in range(1,9)}

def make_windows(arr, window=WINDOW):
    count = len(arr) // window
    if count < 1:
        raise ValueError(f'Not enough samples for {window}-sample windows: {len(arr)}')
    return np.asarray(arr[:count*window].reshape(count,window,arr.shape[1]), dtype=np.float32)

def load_group(h5, name, sensors):
    if name not in h5:
        raise KeyError(f'Missing HDF5 group: {name}. Available: {list(h5.keys())[:15]}')
    acc = h5[name]['acceleration']
    missing = [s for s in sensors if s not in acc or 'y' not in acc[s]]
    if missing:
        raise KeyError(f'{name} lacks required Y channels: {missing}')
    signals = [np.asarray(acc[s]['y'][:], dtype=np.float32).reshape(-1) for s in sensors]
    n = min(map(len, signals))
    data = np.column_stack([x[:n] for x in signals])
    if not np.isfinite(data).all():
        raise ValueError(f'{name} contains NaN or infinity')
    return data

def build_model(n_features):
    # Proposed simple Dense baseline, NOT a confirmed group/supervisor architecture.
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu', name='bottleneck'),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_features, activation='linear')
    ], name='proposed_dense_baseline')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def anomaly_scores(model, x, batch_size=32):
    flat = x.reshape(len(x), -1)
    pred = model.predict(flat, batch_size=batch_size, verbose=0)
    return np.mean(np.square(flat-pred), axis=1)

def run(data_file, out_dir, epochs=50, batch_size=8):
    data_file, out = Path(data_file).expanduser(), Path(out_dir).expanduser()
    if not data_file.is_file():
        raise FileNotFoundError(f'HBTA HDF5 dataset not found: {data_file.resolve()}. Supply --data PATH.')
    out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    with h5py.File(data_file, 'r') as h5:
        if HEALTHY not in h5:
            raise KeyError(f'Missing healthy group {HEALTHY}; available groups: {list(h5.keys())[:20]}')
        acc = h5[HEALTHY]['acceleration']
        sensors = [f'AG{i:02d}' for i in range(1,19)
                   if f'AG{i:02d}' in acc and 'y' in acc[f'AG{i:02d}']]
        if len(sensors) != 17:
            raise ValueError(f'Expected 17 Y channels, found {len(sensors)}: {sensors}')
        healthy = load_group(h5, HEALTHY, sensors)
        reference = load_group(h5, REFERENCE, sensors)
        damaged = {k:load_group(h5,g,sensors) for k,g in DAMAGE.items()}
    n = len(healthy)
    i,j = int(.70*n),int(.85*n)
    train_raw,val_raw,test_raw=healthy[:i],healthy[i:j],healthy[j:]
    scaler = StandardScaler().fit(train_raw)
    x_train,x_val,x_test=[make_windows(scaler.transform(z)) for z in (train_raw,val_raw,test_raw)]
    x_ref=make_windows(scaler.transform(reference))
    x_damage={k:make_windows(scaler.transform(z)) for k,z in damaged.items()}
    print('Sensors:', sensors)
    print('Windows train/validation/test/independent healthy:',*[len(x) for x in (x_train,x_val,x_test,x_ref)])
    print('Damage windows:',{k:len(v) for k,v in x_damage.items()})
    flat=lambda x:x.reshape(len(x),-1)
    model=build_model(WINDOW*len(sensors))
    model.summary()
    history=model.fit(flat(x_train),flat(x_train),validation_data=(flat(x_val),flat(x_val)),
                      epochs=epochs,batch_size=batch_size,shuffle=True,verbose=2,
                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)])
    pd.DataFrame(history.history).to_csv(out/'training_history.csv',index=False)
    plt.figure(figsize=(8,4));plt.plot(history.history['loss'],label='Train');plt.plot(history.history['val_loss'],label='Validation')
    plt.xlabel('Epoch');plt.ylabel('Reconstruction MSE');plt.legend();plt.tight_layout();plt.savefig(out/'training_loss.png',dpi=180);plt.close()
    healthy_scores={'held_out_healthy_test':anomaly_scores(model,x_test),
                    'separate_healthy_recording':anomaly_scores(model,x_ref)}
    damage_scores={k:anomaly_scores(model,x) for k,x in x_damage.items()}
    rows=[]
    for ref_name, ref_scores in healthy_scores.items():
        for name, scores in damage_scores.items():
            labels=np.r_[np.zeros(len(ref_scores)),np.ones(len(scores))]
            values=np.r_[ref_scores,scores]
            rows.append(dict(healthy_reference=ref_name,condition=name,healthy_windows=len(ref_scores),
                             damaged_windows=len(scores),roc_auc=float(roc_auc_score(labels,values)),
                             healthy_mean_mse=float(ref_scores.mean()),damage_mean_mse=float(scores.mean())))
    results=pd.DataFrame(rows)
    results.to_csv(out/'roc_auc_by_condition.csv',index=False)
    print('\nROC-AUC by condition (not classification accuracy):\n',results.to_string(index=False))
    print('\nMean ROC-AUC by reference:\n',results.groupby('healthy_reference')['roc_auc'].mean().to_string())
    fig,ax=plt.subplots(figsize=(9,4))
    for ref_name, subset in results.groupby('healthy_reference'):
        ax.plot(subset['condition'],subset['roc_auc'],marker='o',label=ref_name)
    ax.axhline(.5,ls='--',c='gray',label='Chance ranking')
    ax.set_ylim(0,1);ax.set_ylabel('ROC-AUC');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'roc_auc_chart.png',dpi=180);plt.close(fig)
    validation_scores=anomaly_scores(model,x_val)
    threshold=float(np.percentile(validation_scores,95))
    threshold_rows=[dict(condition='held_out_healthy_test',fraction_flagged=float(np.mean(healthy_scores['held_out_healthy_test']>threshold)))]
    threshold_rows += [dict(condition=k,fraction_flagged=float(np.mean(v>threshold))) for k,v in damage_scores.items()]
    pd.DataFrame(threshold_rows).to_csv(out/'threshold_diagnostics.csv',index=False)
    model.save(out/'proposed_dense_baseline.keras')
    joblib.dump(scaler,out/'training_scaler.joblib')
    pd.DataFrame({'sensor':sensors}).to_csv(out/'sensor_order.csv',index=False)
    config={'model_status':'PROPOSED Dense baseline, NOT confirmed with teammate/supervisor',
            'architecture':[WINDOW*len(sensors),128,32,128,WINDOW*len(sensors)],
            'seed':SEED,'healthy_group':HEALTHY,'reference_group':REFERENCE,'damage_groups':DAMAGE,
            'sampling_hz':100,'window_size':WINDOW,'train_val_test':'70/15/15 chronological on UDS_01',
            'scaler':'StandardScaler fitted on training only','epochs_max':epochs,'batch_size':batch_size,
            'threshold_95pct_validation':threshold,'dataset_path':str(data_file.resolve())}
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('\nOutputs saved in:',out.resolve())
    return results

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='HBTA proposed Dense Autoencoder baseline')
    p.add_argument('--data',default='data_100Hz.h5',help='Path to HBTA HDF5 dataset')
    p.add_argument('--out',default='baseline_outputs',help='Output directory')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    run(a.data,a.out,a.epochs,a.batch_size)
