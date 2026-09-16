CC = gcc

ifeq ($(OS),Windows_NT)
    TARGET = SintensisVoz.exe
    CFLAGS = -O2 -mwindows
else
    TARGET = SintesisVoz
    CFLAGS = -O2
endif

SRC = launcher.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f SintensisVoz.exe SintesisVoz
