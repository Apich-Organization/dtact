CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -L../target/release -ldtact -Wl,-rpath,../target/release

all: c_async

c_async: c_async.c
	$(CC) $(CFLAGS) -o c_async c_async.c $(LDFLAGS)

run: c_async
	./c_async

clean:
	rm -f c_async
