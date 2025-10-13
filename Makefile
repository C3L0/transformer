CC = gcc
CFLAGS = -Wall -O2 -Iinclude

SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:.c=.o)
TARGET = transformer

$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f src/*.o main.o $(TARGET)

