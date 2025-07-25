from pyod.models.alad import ALAD
import math

class ALAD_wrapper(ALAD):
    def __init__(self, n_layers=1, shrinkage_factor=0.3, **args):
        
        self.n_layers = n_layers
        self.shrinkage_factor = shrinkage_factor

        try: 
            del args["encoder_neurons"]
        except KeyError:
            pass
        
        try: 
            del args["decoder_neurons"]
        except KeyError:
            pass
        
        self.args = args
    
    def fit(self, X, y=None):
        
        n_features = X.shape[1]
        
        self.encoder_neurons = [math.ceil(n_features * (1-self.shrinkage_factor)**(i+1)) for i in range(self.n_layers)]
        
        self.decoder_neurons = list(reversed(self.encoder_neurons))
        
        
        super().__init__(dec_layers=self.decoder_neurons, enc_layers=self.encoder_neurons, disc_xx_layers=self.encoder_neurons, disc_zz_layers=self.encoder_neurons, disc_xz_layers=self.encoder_neurons, **self.args)
        
        super().fit(X, y)