target:	fem1d

fem1d:
	g++ -o fem1d SpectralHeat.cpp
	echo "Usage: ./fem1d p element length nu timesteps"


clean:
	rm fem1d solution.txt