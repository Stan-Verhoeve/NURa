# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Iincludes -I/usr/include/hdf5/serial
LDFLAGS = -lm -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5

# Automatically find all .c files in includes/
INCLUDES_SRCS := $(wildcard includes/*.c)
INCLUDES_OBJS := $(patsubst %.c,%.o,$(INCLUDES_SRCS))

# Scripts and their objects
SCRIPT_SRCS := # TODO: Add script .c names here 
SCRIPT_OBJS := $(patsubst %.c,%.o,$(SCRIPT_SRCS))
EXECUTABLES := # TODO: Add script names here


# TODO: Add scripts here in format:
# name_of_script: name_of_script.o $(INCLUDE_OBJS)
#     $(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


# Alternatively, we can do it automatically for all .c files
# Auto-detect .c files in current directory (excluding includes/)
SCRIPT_SRCS := $(filter-out $(notdir $(INCLUDES_SRCS)), $(wildcard *.c))
SCRIPT_OBJS := $(patsubst %.c,%.o,$(SCRIPT_SRCS))
EXECUTABLES := $(patsubst %.c, %, $(SCRIPT_SRCS))

# Default target
all: $(EXECUTABLES)

# Generic rule to build each executable from .o and includes
%: %.o $(INCLUDES_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


# Compile any .c file into a .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f *.o includes/*.o $(EXECUTABLES)

.PHONY: all clean

