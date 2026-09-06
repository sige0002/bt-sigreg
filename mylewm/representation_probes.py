"""Frozen-feature diagnostics. Probe fitting never updates a world model."""
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def task_probe(xs,labels):
    """Diagnostic task readout only; labels never enter the world model."""
    classes=np.unique(labels[0])
    if len(classes)<2 or any(not np.isin(y,classes).all() for y in labels[1:]):
        raise ValueError('Require multiple known training classes')
    scaler=StandardScaler().fit(xs[0]);x=[scaler.transform(v) for v in xs]
    best=None
    with threadpool_limits(limits=4):
        for alpha in (.001,.01,.1,1.,10.,100.):
            model=RidgeClassifier(alpha=alpha).fit(x[0],labels[0])
            score=float(np.mean(model.predict(x[1])==labels[1]))
            if best is None or score>best[0]:best=(score,alpha,model)
        prediction=best[2].predict(x[2])
    return {'alpha':best[1],'validation_accuracy':best[0],
            'test_accuracy':float(np.mean(prediction==labels[2])),
            'test_per_task_accuracy':{str(t):float(np.mean(prediction[labels[2]==t]==t))
                                      for t in np.unique(labels[2])},
            'caveat':'Task identity can be decoded from background; not evidence of control or causal factors.'}


def regression_probe(xs,ys,nonlinear=False):
    """Train-only scaling and fitting; alpha selected on validation, test once.

    This is predictive redundancy, not mutual information or causal identity.
    Constant training targets are excluded and reported explicitly.
    """
    if len(xs)!=3 or len(ys)!=3: raise ValueError('Require train, validation, test')
    if any(not np.isfinite(v).all() for v in [*xs,*ys]): raise ValueError('Nonfinite features/targets')
    scaler=StandardScaler().fit(xs[0])
    x=[scaler.transform(v) for v in xs]
    mean=ys[0].mean(0);std=ys[0].std(0)
    valid=std>1e-8
    if not valid.any(): raise ValueError('All training targets are constant')
    y=[(v[:,valid]-mean[valid])/std[valid] for v in ys]
    with threadpool_limits(limits=4):
        if nonlinear:
            mapping=Nystroem(kernel='rbf',gamma=1./x[0].shape[1],
                             n_components=min(128,len(x[0])),random_state=773)
            x=[mapping.fit_transform(x[0]),mapping.transform(x[1]),mapping.transform(x[2])]
        best=None
        for alpha in (.001,.01,.1,1.,10.,100.):
            model=Ridge(alpha=alpha).fit(x[0],y[0])
            mse=float(np.square(model.predict(x[1]).reshape(y[1].shape)-y[1]).mean())
            if best is None or mse<best[0]: best=(mse,alpha,model)
        pred=best[2].predict(x[2]).reshape(y[2].shape)
    residual=np.square(pred-y[2]).sum(0)
    total=np.square(y[2]-y[2].mean(0)).sum(0)
    test_valid=total>1e-10
    r2=1-residual[test_valid]/total[test_valid]
    return {'kind':'rbf_nystroem128_ridge' if nonlinear else 'linear_ridge',
            'alpha':best[1],'validation_train_scaled_mse':best[0],
            'test_train_scaled_mse':float(np.square(pred-y[2]).mean()),
            'test_r2_mean':float(r2.mean()) if len(r2) else None,
            'test_r2_per_valid_coordinate':r2.tolist(),
            'valid_training_coordinates':np.flatnonzero(valid).tolist(),
            'nonconstant_test_coordinates_within_valid':np.flatnonzero(test_valid).tolist(),
            'training_target_mean':mean.tolist(),'training_target_std':std.tolist()}


def diagnose(zs,states,blocks=4):
    """zs/states each contain train/val/test arrays [trajectory,time,coordinate]."""
    if any(z.ndim!=3 or z.shape[1]<2 for z in zs): raise ValueError('Need real temporal windows')
    if zs[0].shape[-1]%blocks: raise ValueError('Unequal blocks')
    full=[z.reshape(-1,z.shape[-1]) for z in zs]
    state=[s.reshape(-1,s.shape[-1]) for s in states]
    dz=[np.diff(z,axis=1).reshape(-1,z.shape[-1]) for z in zs]
    ds=[np.diff(s,axis=1).reshape(-1,s.shape[-1]) for s in states]
    means=[z.mean(1) for z in zs];state_means=[s.mean(1) for s in states]
    readout={}
    for block in [None,*range(blocks)]:
        name='all' if block is None else f'block_{block}'
        select=lambda x:x if block is None else np.split(x,blocks,axis=-1)[block]
        readout[name]={label:regression_probe([select(x) for x in source],target)
                      for label,source,target in [('instant_state',full,state),
                          ('observed_state_change',dz,ds),('window_mean_state',means,state_means)]}
    redundancy={}
    for i in range(blocks):
        for j in range(blocks):
            if i==j:continue
            xs=[np.split(x,blocks,axis=-1)[i] for x in full]
            ys=[np.split(x,blocks,axis=-1)[j] for x in full]
            redundancy[f'{i}_to_{j}']={kind:regression_probe(xs,ys,nonlinear=kind=='nonlinear')
                                      for kind in ('linear','nonlinear')}
    test=zs[2]
    return {'state_readouts':readout,'block_predictability':redundancy,
            'test_within_window_variance':float(np.square(test-test.mean(1,keepdims=True)).mean()),
            'test_between_window_mean_variance':float(test.mean(1).var(0).mean()),
            'caveat':'Window means/changes are static/dynamic proxies, not identified physical factors. '
                     'Observed change readout sees both frames; it is not future prediction. '
                     'Nonlinear predictability is not proof of useful physical relations.'}
