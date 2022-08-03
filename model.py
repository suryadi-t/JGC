import numpy as np
import tensorflow as tf
from tensorflow import keras
from functools import reduce, partial
import multiprocessing as mp
from preprocessing import preprocess_data
from sklearn.covariance import MinCovDet

class GateLayer(keras.layers.Layer):
    """A layer used for feature selection."""
    
    def __init__(self):
        super(GateLayer, self).__init__()

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim,),
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.kernel 

def JGC_model(inpt_dim:int, hidden_widths:tuple):
    """Generate neural network model. Hidden widths should be a tuple, where each element is the width of the
    corresponding hidden layer."""
    inp = keras.layers.Input(inpt_dim)
    gate = GateLayer()
    item = gate(inp)
    for width in hidden_widths:
        item = keras.layers.Dense(width,activation='relu')(item)
    out = keras.layers.Dense(1,activation='linear')(item)        
    model = keras.Model(inputs=inp,outputs=out)
    model.compile(optimizer='adam',loss='mse',metrics=['MeanSquaredError'])
    return model

def infer_GC_one_parallel(target_index:int, model, data:np.ndarray, lam:float):
    """for parallelization"""
    X,y = preprocess_data(data,target_index,model.eta)
    grads, var_scores, lag_scores = [],[],[]
    for i in range(model.n_iters):
        grad, var_score, lag_score = model.train_new(X,y,lam,target_index,seed=i)
        grads.append(grad)
        var_scores.append(var_score)
        lag_scores.append(lag_score)
    var_score = var_scores[0] #take scores from first iteration
    lag_score = lag_scores[0] 
    lag_binary, var_binary = model.var_selection_one_target(target_index)
    return var_binary, var_score, lag_binary, lag_score, grads, var_scores, lag_scores

def infer_GC_all_parallel(model, data:np.ndarray, lam:float,processes=8):
    var_binary_mat = np.zeros((model.dim,model.dim)) #[effect,cause]
    var_score_mat = np.zeros((model.dim,model.dim))
    lag_binary_mat = np.zeros((model.dim,model.eta,model.dim)) #[effect,lag,cause]
    lag_score_mat = np.zeros((model.dim,model.eta,model.dim))
    #note that lag is indexed as [eta,eta-1,...,2,1]
    
    infer_one = partial(infer_GC_one_parallel,model=model,data=data,lam=lam)
    
    chunk_size = max(model.dim//processes,1)
    with mp.Pool(processes=processes) as pool:
        results = pool.map(infer_one, range(model.dim), chunk_size)
    
    for target_index in range(model.dim):
        var_binary, var_score, lag_binary, lag_score, grads, var_scores, lag_scores = results[target_index]
        var_binary_mat[target_index] = var_binary
        lag_binary_mat[target_index] = lag_binary
        var_score_mat[target_index] = var_score
        lag_score_mat[target_index] = lag_score
        model.grads[target_index] = grads
        model.lag_scores[target_index] = lag_scores
        model.var_scores[target_index] = var_scores
    model.var_binary = var_binary_mat
    model.lag_binary = lag_binary_mat
    return var_binary_mat, var_score_mat, lag_binary_mat, lag_score_mat

class JGC():
    def __init__(self, dim:int, eta:int, n_iters=3, n_epochs=2000, batch_size=64,
                 hidden_widths=(50,50), sd_cutoff=1.):
        self.dim = dim
        self.eta = eta
        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_widths = hidden_widths
        self.grads = dict()
        self.lag_scores = dict()
        self.var_scores = dict()
        self.sd_cutoff = sd_cutoff #reject variables whose importance score < sd_cutoff*sd away from Gaussian mean
        
    def train_new(self, X:np.ndarray, y:np.ndarray, lam:float, target_index:int, seed=1):
        tf.random.set_seed(seed)
        model = JGC_model(X.shape[1], hidden_widths=self.hidden_widths)
        
        #warm start without L1 regularization
        model.fit(X,y,batch_size=self.batch_size,epochs=int(self.n_epochs/2),verbose=0)
        
        #add L1 regularization on gate layer for sparsity 
        model.layers[1].add_loss(lambda: tf.reduce_sum(tf.abs(model.layers[1].weights))*lam)
        #add L2 regularization with fixed lambda = 0.01 on subsequent layers to prevent overfitting
        for layer_num in range(2,len(model.layers)):
            model.layers[layer_num].add_loss(lambda: tf.reduce_sum(tf.square(model.layers[layer_num].weights[0]))*0.01)
            #for the last index, 0 gives weights and 1 gives biases
        
        model.compile(optimizer='adam',loss='mse',metrics=['MeanSquaredError']) 
        model.fit(X,y,batch_size=self.batch_size,epochs=int(self.n_epochs/2),verbose=0)
        
        tensor = tf.convert_to_tensor(X,dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            out = model(tensor)
        grad = tape.gradient(out,tensor).numpy()
        impt_score = np.abs(np.mean(grad,axis=0))
        var_score, lag_score = self.sum_zero_one_lag(impt_score,target_index)
        self.grads[target_index] = self.grads.get(target_index,[]) + [grad]
        self.lag_scores[target_index] = self.lag_scores.get(target_index,[]) + [lag_score]
        self.var_scores[target_index] = self.var_scores.get(target_index,[]) + [var_score]
        return grad, var_score, lag_score
    
    def sum_zero_one_lag(self, impt_score:np.ndarray, target_index:int):
        """sum the impt_scores for time t and t-1"""
        impt_score = np.insert(impt_score,self.dim*(self.eta) + target_index, 0) #for reshaping 
        impt_score = impt_score.reshape((self.eta+1,self.dim)) #[lag,variable]
        impt_score[-2] = impt_score[-1] + impt_score[-2] #sum lag 0 and lag 1
        lag_score = np.delete(impt_score,self.eta,axis=0) #remove lag 0
        var_score = np.max(lag_score,axis=0) #score for variable (ignoring lag)
        return var_score, lag_score
    
    def significance_test(self,signed_scores:list, sd_cutoff:float, assume_centered=False):
        if assume_centered: 
            # then we cannot use zscore, as depending on the fraction and strength of true signals,
            # the 0 of the zscore may be very far from the cluster of insignificant scores.
            # e.g. with very strong signals we can have zscore=0 near a weak signal and far from
            # the insignificant scores.
            # assume_centered means we expect the insignificant scores to have mean = 0, so 
            # we can't take zscores.
            fitscore = signed_scores
        else: #compute zscore, so the overall magnitude is larger
            #if magnitudes are too small, MinCovDet can have issues where the fit covariance is too small
            #and is taken to be 0, leading to an error
            #we compute zscore manually so we can get the moments to reverse the transform later
            mean_score, sd_score = np.mean(signed_scores), np.std(signed_scores)
            fitscore = (signed_scores-mean_score)/sd_score
        computed=False
        support_fraction=None
        while not computed:
            try:
                if support_fraction==None:
                    cov = MinCovDet(random_state=1,assume_centered=assume_centered).fit(fitscore.reshape(-1,1))
                else:
                    cov = MinCovDet(random_state=1,assume_centered=assume_centered,
                                    support_fraction=support_fraction).fit(fitscore.reshape(-1,1))
                computed = True
            except:
                if support_fraction==1:
                    raise RuntimeError('MinCovDet is unable to fit the importance scores.')
                if support_fraction==None:
                    n_samples = len(fitscore)
                    n_features = 1 #the features are the zscores of the signed importance scores
                    frac = int(np.ceil(0.5 * (n_samples + n_features + 1))) / len(fitscore)
                    
                    #round to next 0.05
                    support_fraction = np.ceil(frac*20)/20
                else: #computed support fraction still fails, so we keep increasing
                    support_fraction += 0.05
                    
        mu,var = cov.location_.flatten(), cov.covariance_.flatten()
        sd = np.sqrt(var)
        #compute left & right thresholds in actual scale (not z-scores)
        left,right = mu - sd_cutoff*sd, mu + sd_cutoff*sd
        
        ## Taking the inferred Gaussian as a null distribution,
        # we remove those whose zscores are inside the left and right cutoffs
        # we ALSO remove those whose signed scores (NOT zscores) are between zero
        # and the left-right limits, as those would be even less significant
        # than the removed ones, and are therefore also insignificant.
        # Note: for this we have to use the actual scores instead of zscores
        
        cutoff = np.where(np.logical_and(fitscore>left,fitscore<right))[0]
        
        if assume_centered: #no change
            abs_left,abs_right = left,right
        else: #reverse the z-score transform to get left right in actual scale
            abs_left,abs_right = left*sd_score + mean_score, right*sd_score + mean_score
        
        if abs_left<0 and abs_right<0:
            weaker_vars = np.where(np.logical_and(signed_scores<0,signed_scores>abs_left))[0]
        elif abs_left>0 and abs_right>0:
            weaker_vars = np.where(np.logical_and(signed_scores>0,signed_scores<abs_right))[0]
        else: #the other possibility is abs_left<0 and abs_right>0, i.e. the cutoff contains all insignificant vars
            weaker_vars = []
        zero_ind = np.union1d(cutoff,weaker_vars)        
        
        abs_score = np.abs(signed_scores)
        sorted_ind = np.argsort(abs_score)[::-1] #in descending order
        sorted_sig_ind = np.array([sorted_ind[i] for i in range(len(sorted_ind)) if sorted_ind[i] not in zero_ind])
        return sorted_sig_ind
        
    def consistency_test(self, ordered:list):
        """Note: expects score indices sorted in descending importance score"""
        inters = reduce(np.intersect1d,ordered)
        converged = False
        while not converged: 
            for iteration in range(len(ordered)):
                for k in np.arange(len(ordered[iteration])):
                    if ordered[iteration][k] not in inters:
                        ordered[iteration] = ordered[iteration][:k]
                        break
            new_inters = reduce(np.intersect1d,ordered)
            if list(inters)==list(new_inters): converged = True
            inters = new_inters #continue trimming
            if len(inters)==0: converged = True
        accept = reduce(np.intersect1d,ordered)
        return accept
    
    def var_selection_one_target(self, target_index:int):
        sig_ind_list = list()
        for iteration in range(3):
            _, lag_sign = self.get_sign_one_target(target_index,iteration=iteration,threshold=False)
            signed_scores = (self.lag_scores[target_index][iteration] * lag_sign).flatten()
            sig_ind_list.append(self.significance_test(signed_scores,self.sd_cutoff))
        accepted = self.consistency_test(sig_ind_list)
        
        onehot = np.zeros((self.eta*self.dim))
        onehot[accepted] = 1
        lag_binary = onehot.reshape((self.eta,self.dim))
        var_binary = np.max(lag_binary,axis=0)
        return lag_binary, var_binary
        
    def infer_GC_one_target(self, data:np.ndarray, target_index:int, lam:float, verbose=True):
        X,y = preprocess_data(data,target_index,self.eta)
        if verbose: print('..running iteration ',end='')
        for i in range(self.n_iters):
             if verbose: print('%d/%d'%(i+1,self.n_iters),end=' ')
             self.train_new(X,y,lam,target_index,seed=i)
        var_score = self.var_scores[target_index][0] #take scores from first iteration
        lag_score = self.lag_scores[target_index][0] 
        lag_binary, var_binary = self.var_selection_one_target(target_index)
        return var_binary, var_score, lag_binary, lag_score
    
    def infer_GC_all(self, data:np.ndarray, lam:float, verbose=True, parallelize=False, processes=8):
        if parallelize:
            return infer_GC_all_parallel(self, data, lam, processes=processes)
        var_binary_mat = np.zeros((self.dim,self.dim)) #[effect,cause]
        var_score_mat = np.zeros((self.dim,self.dim))
        lag_binary_mat = np.zeros((self.dim,self.eta,self.dim)) #[effect,lag,cause]
        lag_score_mat = np.zeros((self.dim,self.eta,self.dim))
        #note that lag is indexed as [eta,eta-1,...,2,1]
        for target_index in range(self.dim):
            if verbose: print('\nRunning for target index %d/%d'%(target_index,self.dim-1))
            var_binary, var_score, lag_binary, lag_score = self.infer_GC_one_target(data,target_index,lam,verbose=verbose)
            var_binary_mat[target_index] = var_binary
            lag_binary_mat[target_index] = lag_binary
            var_score_mat[target_index] = var_score
            lag_score_mat[target_index] = lag_score
        self.var_binary = var_binary_mat
        self.lag_binary = lag_binary_mat
        return var_binary_mat, var_score_mat, lag_binary_mat, lag_score_mat
    
    def get_sign_one_target(self, target_index:int, iteration=0, threshold=False):
        if target_index not in self.grads:
            print('No result for target index. Run JGC.infer_GC_one_target() first.')
            return
        meangrad = np.mean(self.grads[target_index][iteration],axis=0)
        var_score, lag_score = self.sum_zero_one_lag(meangrad,target_index)
        var_sign, lag_sign = np.sign(var_score), np.sign(lag_score)
        if threshold: #set irrelevant variables to 0
            var_sign = var_sign * self.var_binary[target_index]
            lag_sign = lag_sign * self.lag_binary[target_index]
        return var_sign, lag_sign
        
    def get_sign_all(self, threshold=False):
        var_sign_mat = np.zeros((self.dim,self.dim))
        lag_sign_mat = np.zeros((self.dim,self.eta,self.dim))
        for target_index in range(self.dim):
            var_sign, lag_sign = self.get_sign_one_target(target_index,threshold=threshold)
            var_sign_mat[target_index] = var_sign
            lag_sign_mat[target_index] = lag_sign
        return var_sign_mat, lag_sign_mat
        
