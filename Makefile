target:	fem1d

fem1d: SpectralHeat.o


SpectralHeat.o: SpectralHeat.cpp
	g++ -std=c++11 -o fem1d SpectralHeat.cpp
	echo "Usage: ./fem1d p element length nu timesteps"


clean:
	rm fem1d solution.png elapsed_time.txt solution.txt