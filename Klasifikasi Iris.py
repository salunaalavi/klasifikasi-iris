# Mengimport library yang dibutuhkan
from csv import reader
from math import sqrt
from math import exp
from math import pi
from random import seed
from random import randrange
 
# Membaca dataset Iris
def baca_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for baris in csv_reader:
			if not baris:
				continue
			dataset.append(baris)
	return dataset
 
# Konversi kolom dengan data string menjadi float
def konv_kolom_str_ke_float(dataset, kolom):
	for baris in dataset:
		baris[kolom] = float(baris[kolom].strip())
 
# Konversi kolom dengan data string menjadi integer
def konv_kolom_str_ke_int(dataset, kolom):
	data_kelas = [baris[kolom] for baris in dataset]
	unique = set(data_kelas)
	cari_string = dict()
	for i, data in enumerate(unique):
		cari_string[data] = i
	for baris in dataset:
		baris[kolom] = cari_string[baris[kolom]]
	return (cari_string, data_kelas)
 
# Pisahkan dataset berdasarkan kelas, hasilnya berupa dictionary(dict())
def pisah_kelas(dataset):
	pisah = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		data_kelas = vector[-1]
		if (data_kelas not in pisah):
			pisah[data_kelas] = list()
		pisah[data_kelas].append(vector)
	return pisah
 
# Hitung rata-rata
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Hitung standar deviasi
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Hitung rata-rata, standar deviasi dari setiap kolom di dalam dataset
def hitung_dataset(dataset):
	hasil_dataset = [(mean(kolom), stdev(kolom), len(kolom)) for kolom in zip(*dataset)]
	del(hasil_dataset[-1])
	return hasil_dataset

# Pisahkan dataset berdasarkan kelas untuk menghitung statistik dari setiap baris
def hitung_kelas(dataset):
	pisah = pisah_kelas(dataset)
	hasil_kelas = dict()
	for data_kelas, baris in pisah.items():
		hasil_kelas[data_kelas] = hitung_dataset(baris)
	return hasil_kelas

# Hitung fungsi distribusi probabilitas Gaussian
def hitung_probabilitas(x, mean, stdev):
	eksponen = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * eksponen

# Hitung probabilitas data baru terhadap data setiap kelas
def hitung_probabilitas_kelas(model, baris):
	total_baris = sum([model[label][0][2] for label in model])
	probabilitas = dict()
	for data_kelas, hasil_setiap_kelas in model.items():
		probabilitas[data_kelas] = model[data_kelas][0][2]/float(total_baris)
		for i in range(len(hasil_setiap_kelas)):
			mean, stdev, _ = hasil_setiap_kelas[i]
			probabilitas[data_kelas] *= hitung_probabilitas(baris[i], mean, stdev)
	return probabilitas

# Prediksi kelas untuk nilai yang diinputkan
def prediksi(model, baris):
	probabilitas_kelas = hitung_probabilitas_kelas(model, baris)
	best_label, best_prob = None, -1
	for data_kelas, probability in probabilitas_kelas.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = data_kelas
	return best_label

# Bagi dataset ke suatu bentuk k-folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Hitung persentase akurasi
def metric_akurasi(aktual, prediksi):
	hasil = 0
	for i in range(len(aktual)):
		if aktual[i] == prediksi[i]:
			hasil += 1
	return hasil / float(len(aktual)) * 100.0

# Algoritma evaluasi menggunakan cross validation split
def algoritma_evaluasi(dataset, model_nb, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	hasil = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for baris in fold:
			baris_copy = list(baris)
			test_set.append(baris_copy)
			baris_copy[-1] = None
		prediksi = model_nb(train_set, test_set, *args)
		aktual = [baris[-1] for baris in fold]
		akurasi = metric_akurasi(aktual, prediksi)
		hasil.append(akurasi)
	return hasil

# Algoritma Naive Bayes untuk menghitung tingkat akurasi
def naive_bayes(train, test):
	data = hitung_kelas(train)
	nilai_prediksi = list()
	for baris in test:
		output = prediksi(data, baris)
		nilai_prediksi.append(output)
	return nilai_prediksi

# Deklarasi dataset
def dataset_iris():
    filename = 'Dataset Iris.csv'
    dataset = baca_csv(filename)
    for i in range(len(dataset[0])-1):
    	konv_kolom_str_ke_float(dataset, i)
    konv_kolom_str_ke_int(dataset, len(dataset[0])-1)
    return dataset


# Input dinamis
def do_input():
    dataset = dataset_iris()
    # Fit model
    model = hitung_kelas(dataset)
    # Input data baru
    sepal_length = float(input('Panjang Sepal (Float): '))
    sepal_width = float(input('Lebar Sepal (Float): '))
    petal_length = float(input('Panjang Petal (Float): '))
    petal_width = float(input('Lebar Petal (Float): '))
    
    label = prediksi(model, [sepal_length, sepal_width, petal_length, petal_width])
    print('\nTumbuhan terklasifikasi sebagai: ')
    if label == 0:
        print('Iris-Setosa')
    elif label == 1:
        print('Iris-Virginica')
    else:
        print('Iris-Versicolor')
    print('\n\n')
    
def get_acc():
    dataset = dataset_iris()
    # Tingkat akurasi
    seed(1)
    n_folds = 5
    nilai_hasil = algoritma_evaluasi(dataset, naive_bayes, n_folds)
    print('Nilai akurasi dari %s data split: %s' %(n_folds, nilai_hasil))
    return (sum(nilai_hasil)/float(len(nilai_hasil)))

def show_menu():
    print('[1] Klasifikasi Tumbuhan Iris')
    print('[2] Metric Akurasi')
    print('[3] Exit')
    ans = int(input('Pilih : '))
    
    if ans == 1:
        do_input()
        return 1
    elif ans == 2:
        print(f'Rerata akurasi dengan menggunakan metode cross-validation split: {get_acc()}\n')
        print('\n\n')
        return 1
    else:
        return 0

stop = 1
while stop == 1:
    stop = show_menu()

''' 
# Input statis
filename = 'Dataset Iris.csv'
dataset = baca_csv(filename)
for i in range(len(dataset[0])-1):
	konv_kolom_str_ke_float(dataset, i)
konv_kolom_str_ke_int(dataset, len(dataset[0])-1)
# Fit model
model = hitung_kelas(dataset)
# Data baru
baris = [4.7,3.2,1.3,0.2]
# Prediksi
label = prediksi(model, baris)
print('Data=%s, Prediksi: %s' % (baris, label))


# Tingkat akurasi
seed(1)
n_folds = 5
nilai_hasil = algoritma_evaluasi(dataset, naive_bayes, n_folds)
print('nilai_hasil: %s' % nilai_hasil)
print('Mean akurasi: %.3f%%' % (sum(nilai_hasil)/float(len(nilai_hasil))))
'''