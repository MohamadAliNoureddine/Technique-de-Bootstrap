import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
saved_model_directory_Standard= "/home/noureddine/codes/efispec3d.git/efispec3d-interne/test/Dumanoir/Neural-Network/StandardNN"
loaded_model_Standard = load_model(saved_model_directory_Standard)
freq_min=0
freq_max=0.3333333e+03
num_freq_points=16385
frequences=np.linspace(freq_min,freq_max,num_freq_points)

def model_standard(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie_standard=loaded_model_Standard.predict(inputs)
    return sortie_standard

######### Plan d'experience pour l'analyse de sensibilité globale en utilisant la suite de Sobol
from SALib.sample import saltelli
nombre_lignes =4096
nombre_colonnes = 12
problem = {
    'num_vars': 12,
    'names': [f'input_{i}' for i in range(12)],
    'bounds': [[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520],[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520]]
}
Uncertain_parameters= saltelli.sample(problem, nombre_lignes)
min_colon1= np.min(Uncertain_parameters[:, 0])
max_colon1=np.max(Uncertain_parameters[:, 0])
X11 = (Uncertain_parameters[:, 0] - min_colon1) / (max_colon1 - min_colon1)
min_colon2= np.min(Uncertain_parameters[:, 1])
max_colon2=np.max(Uncertain_parameters[:, 1])
X21= (Uncertain_parameters[:, 1] - min_colon2) / (max_colon2- min_colon2)
min_colon3= np.min(Uncertain_parameters[:, 2])
max_colon3=np.max(Uncertain_parameters[:, 2])
X31= (Uncertain_parameters[:, 2] - min_colon3) / (max_colon3 - min_colon3)
min_colon4= np.min(Uncertain_parameters[:, 3])
max_colon4=np.max(Uncertain_parameters[:, 3])
X41 = (Uncertain_parameters[:, 3] - min_colon4) / (max_colon4 - min_colon4)
min_colon5= np.min(Uncertain_parameters[:, 4])
max_colon5=np.max(Uncertain_parameters[:, 4])
X51 = (Uncertain_parameters[:, 4] - min_colon5) / (max_colon5- min_colon5)
min_colon6= np.min(Uncertain_parameters[:, 5])
max_colon6=np.max(Uncertain_parameters[:, 5])
X61 = (Uncertain_parameters[:, 5] - min_colon6) / (max_colon6 - min_colon6)
min_colon7= np.min(Uncertain_parameters[:, 6])
max_colon7=np.max(Uncertain_parameters[:, 6])
X12 = (Uncertain_parameters[:, 6] - min_colon7) / (max_colon7 - min_colon7)
min_colon8= np.min(Uncertain_parameters[:, 7])
max_colon8=np.max(Uncertain_parameters[:, 7])
X22= (Uncertain_parameters[:, 7] - min_colon8) / (max_colon8- min_colon8)
min_colon9= np.min(Uncertain_parameters[:, 8])
max_colon9=np.max(Uncertain_parameters[:, 8])
X32= (Uncertain_parameters[:, 8] - min_colon9) / (max_colon9 - min_colon9)
min_colon10= np.min(Uncertain_parameters[:, 9])
max_colon10=np.max(Uncertain_parameters[:, 9])
X42 = (Uncertain_parameters[:, 9] - min_colon10) / (max_colon10- min_colon10)
min_colon11= np.min(Uncertain_parameters[:, 10])
max_colon11=np.max(Uncertain_parameters[:, 10])
X52= (Uncertain_parameters[:, 10] - min_colon11) / (max_colon11- min_colon11)
min_colon12= np.min(Uncertain_parameters[:, 11])
max_colon12=np.max(Uncertain_parameters[:, 11])
X62= (Uncertain_parameters[:, 11] - min_colon12) / (max_colon12 - min_colon12)
Output = model_standard(X11,X21,X31,X41,X51,X61)
############################Méthode de Li et mahadevan
X_in=Uncertain_parameters[:, :6]
y_in =Output
def compute_sobol_index(x, y, ninter, nminvar):
    min_x = np.min(x)
    max_x = np.max(x)
    dx = (max_x - min_x) / ninter
    pi_hist = np.linspace(min_x, max_x, ninter + 1)
    VectVariancePI = np.zeros(ninter)
    for i in range(ninter):
        mask = np.logical_and(x >= pi_hist[i], x <= pi_hist[i + 1])
        nmask = np.count_nonzero(mask)
        if nmask > nminvar:
            y_mask = y[mask]
            print(f"Interval {i}:")
            print(y_mask.shape)
            VectVariancePI[i] = np.var(y_mask)
        else:
            print('Erreur : Nombre insuffisant de points pour calculer la variance')
    S = 1 - np.mean(VectVariancePI) / np.var(y)
    return S
n_params = X_in.shape[1]  # Nombre de paramètres incertains
n_times = y_in.shape[1]   # Nombre de pas de temps dans les sorties
sobol_indices = np.zeros((n_params, n_times))  # Matrice pour stocker les indices de Sobol
for param in range(n_params):  # Itération sur chaque paramètre incertain
    x = X_in[:, param]  # Sélection du paramètre
    for time_step in range(n_times):  # Itération sur chaque pas de temps
        y = y_in[:, time_step]  # Données pour ce pas de temps
        # Calcul des indices de Sobol pour ce paramètre à ce pas de temps
        Sobol_index = compute_sobol_index(x, y, ninter=30,nminvar=20)
        # Stockage de l'indice de Sobol pour ce paramètre à ce pas de temps
        sobol_indices[param, time_step] = Sobol_index
plt.figure(figsize=(10, 8))
cumulative_influence = np.zeros_like(sobol_indices[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
for param in range(n_params):
    influence = sobol_indices[param]
    plt.fill_between(frequences, cumulative_influence, cumulative_influence + influence, alpha=0.3,label=noms_parametres[param])
    cumulative_influence += influence
plt.title('Méthode de Li et Mahadevan')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de sobol cumulées ')
plt.xlim(0, 1.3)
plt.legend()
plt.show()

######################Bootstrap
def calculate_sobol_indices_bootstrap(X_in, y_in, ninter, nminvar, num_bootstrap_samples):
    n_params = X_in.shape[1]
    n_times = y_in.shape[1]
    sobol_indices_bootstrap = np.zeros((n_params, n_times, num_bootstrap_samples))
    for bootstrap_sample in range(num_bootstrap_samples):
        indices_bootstrap = np.random.choice(np.arange(y_in[:,0].shape[0]), size=y_in[:,0].shape[0], replace=True)
        for param in range(n_params):
            x = X_in[indices_bootstrap, param]
            for time_step in range(n_times):
                y = y_in[indices_bootstrap, time_step]
                Sobol_index = compute_sobol_index(x, y, ninter, nminvar)
                sobol_indices_bootstrap[param, time_step, bootstrap_sample] = Sobol_index
    return sobol_indices_bootstrap
num_bootstrap_samples = 20
sobol_indices_bootstrap = calculate_sobol_indices_bootstrap(X_in, y_in, ninter=30, nminvar=20, num_bootstrap_samples=num_bootstrap_samples)

# Visualisation de l'indice de Sobol pour Vs1 sur chaque échantillon bootstrap
plt.figure(figsize=(12, 8))
for i in range (num_bootstrap_samples):
     plt.plot(frequences, sobol_indices_bootstrap[0,:,i])
     plt.title(f'Indices de Sobol pour Vs1 à chaque-Bootstrap sample {i+1}')
     plt.xlabel('Fréquences')
     plt.xlim(0,1.3)
     plt.ylabel('Indice de Sobol')
     plt.legend()
     plt.show()
bootstrap_sobol_array = np.array(sobol_indices_bootstrap)
mean_des_indices=np.mean(sobol_indices_bootstrap,axis=2)
variance_des_indices=np.var(sobol_indices_bootstrap,axis=2)
ecart_type_des_indices=np.std(sobol_indices_bootstrap,axis=2)
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 18))
ax1.plot(frequences, mean_des_indices[0], color='green', linewidth=2,label="Mean des indices")
ax1.set_xlim(0, 1.3)
ax1.set_xlabel('Frequency[Hz]')
ax1.set_ylabel('indices de sobol')
ax1.legend()
ax2.plot(frequences, variance_des_indices[0], color='blue', linewidth=2)
ax2.set_xlim(0, 1.3)
ax2.set_xlabel('Frequency[Hz]')
ax2.set_ylabel('Variance ')
ax2.legend()
ax3.plot(frequences, mean_des_indices[0], color='green', linewidth=2)
ax3.fill_between(frequences, mean_des_indices[0] + 3*ecart_type_des_indices[0], mean_des_indices[0] - 3*ecart_type_des_indices[0], color='gray', alpha=0.5, label='mean of means ± 3*standard deviation')
ax3.set_xlim(0, 1.3)
ax3.set_xlabel('Frequency[Hz]')
ax3.set_ylabel('Indices de sobol')
ax3.legend()
plt.tight_layout()
plt.show()