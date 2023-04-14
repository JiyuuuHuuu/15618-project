SRC=firework.cpp
OBJ=firework.o
CC=g++
CFLAGS=-lglut -lGLU -lGL

firework: $(OBJ)
	$(CC) $(OBJ) -o firework $(CFLAGS)

$(OBJ): $(SRC)
	$(CC) -c $(SRC)

clean:
	rm -f *.o firework
