from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.pylab import plt
from matplotlib import colors as mcolors

#--------------------------------------------
def plot_data(X, Y=None, transform="PCA"):
	color_names = ['r', 'g', 'b', 'm', 'c', 'y', 'k'] + list( mcolors.CSS4_COLORS.keys() )
	XX = PCA(n_components=2).fit_transform(X) if transform=="PCA" else TSNE(n_components=2).fit_transform(X)
	if Y is None: plt.scatter( *zip(*XX) )
	else: plt.scatter( *zip(*XX), c=[color_names[y] for y in Y] )
	plt.show()
	
#--------------------------------------------
lines = open("pendigits.tra").readlines()
DATA = []
#for i in range(0,16):
for line in lines:
        x = [ int(v) for v in line.replace(" ", "").strip().split(",") ]
        #temp=x[:i+2][i:]
        DATA.append(x)

#--------------------------------------------
km = KMeans( n_clusters = 10 )
Y = km.fit_predict( DATA )

#--------------------------------------------
plot_data(DATA, Y)
