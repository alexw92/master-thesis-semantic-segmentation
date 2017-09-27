# Tensorflow Framework

## Knowledge & Tutorials
+ Nice TensorFlow tutorial on [youtube](https://www.youtube.com/watch?v=yX8KuPZCAMo&lc=z12vwxxpxxqmjfwbv04cglyxixiuxrxp0aw) (about 50 min)

## Tensorflow installieren
Install Python 3.5.2 and include the installed folder into Path  (```C:\Wichtige Progs\Python``` in my case)

If you want to use a Nvidia Graphics card to run the the code refer to [this](http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html).

Install with ```pip``` according to the [docs](https://www.tensorflow.org/install/install_windows#installing_with_native_pip)

run the following testCode:

```python
import tensorflow as tf


def main():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


# Call Main method
main()

```

# Python language knowledge

+ Why you need a ```self``` keyword in Python ? [Stackoverflow](https://stackoverflow.com/a/2725996)

+ There are no multiline comments in python apart from using triple ``` ''' multiline comment '''```
which isn't highlighted correctly

+ Python's [with statement](http://effbot.org/zone/python-with-statement.htm)

+ In general, in a statement such as ```Python
class Foo(Base1, Base2):```
Foo is being declared as a class inheriting from base classes Base1 and Base2.

+ [Python 3 Tutorial](https://docs.python.org/3/tutorial/)

+ Print() Knowledge

If you don’t want characters prefaced by \ to be interpreted as special characters, you can use raw strings by adding an r before the first quote:
```Python
>>> print('C:\some\name')  # here \n means newline!
C:\some
ame
>>> print(r'C:\some\name')  # note the r before the quote
C:\some\name
```
Strings can be concatenated (glued together) with the + operator, and repeated with *:

```Python
>>> # 3 times 'un', followed by 'ium'
>>> 3 * 'un' + 'ium'
'unununium'
```

Break long strings (in Java you'd need "abcd"+"abc"+...):

```Python
>>> text = ('Put several strings within parentheses '
...         'to have them joined together.')
>>> text
'Put several strings within parentheses to have them joined together.'
```

Length of a string and cutting:
```Python
str = "abcde"
len( str ) #5
str[3]     #d
str[4]     #e

str[len(str)] #e
str[-1]       #e [Count from the end backwards]

lol = "Alexander"
lol[0:3]     #Ale
lol[:3]      #Ale
lol[3:]      #xander
```

Python string are *immutable*:
```Python
>>> word[0] = 'J'
  ...
TypeError: 'str' object does not support item assignment
```
___
### Lists

```Python
# Indexing like strings
numbers = [1,2,3,4,5]
numbers = numbers + [6,7,8,9]
>>> numbers[0]    # 1 indexing returns the item
>>> numbers[-3:]  # [3,4,5] slicing returns a new list

# Unlike strings, lists are a mutable type
cubes = [1, 8, 27, 65, 125]
cubes[3] = 64
>>> cubes              # [1, 8, 27, 64, 125]
cubes.append(216)  # add the cube of 6
>>> cubes              ## [1, 8, 27, 64, 125, 216]

# Slizing is also possible
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> letters # ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# replace some values
>>> letters[2:5] = ['C', 'D', 'E']
>>> letters # ['a', 'b', 'C', 'D', 'E', 'f', 'g']
# now remove them
>>> letters[2:5] = []
>>> letters # ['a', 'b', 'f', 'g']
# clear the list by replacing all the elements with an empty list
letters[:] = []
>>> letters # []
# len() works with lists as well
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> len(letters)  # 3
```

### Nested lists
```Python
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
>>> x  # [['a', 'b', 'c'], [1, 2, 3]]
>>> x[0] # ['a', 'b', 'c']
>>> x[0][1] # 'b'

```

___
### Control Structures

Fibonacci example:
```Python
>>> a, b = 0, 1
>>> while b < 1000:
...     print(b, end=',')
...     a, b = b, a+b
```

if-statement:
```Python
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')
```

for-loop:
```Python
# Iterating over a copy in order to keep loop range
for w in words[:]:  # Loop over a slice copy of the entire list.
    if len(w) > 6:
        words.insert(0, w)
        
# loop over range
for i in range(5):
    print(i)      # prints numbers from 0 to 4

range(5, 10)      # range from 5 to 9

range(0, 10, 3)   # 0 to 9 in 3-steps: 0, 3, 6, 9

# Iterate with indices
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
    print(i, a[i])
    
# Create List from range (Matlab style :D)
list(range(5)) # [0, 1, 2, 3, 4, 4]

# else in for-loop
# else is entered if loop range is exhausted without break 
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
```

The *pass* statement does nothing. It can be used when a statement is required syntactically but the program requires no action. **Remember:** There are no semicolons ";" in python. 
```Python
while True:
    pass  # Busy-wait for keyboard interrupt (Ctrl+C)

class MyEmptyClass:
    pass
    
def initlog(*args):
    pass   # Remember to implement this!
```

___

Defning functions:
```Python
>>> def fib2(n):  # return Fibonacci series up to n
...     """Return a list containing the Fibonacci series up to n."""
...     result = []
...     a, b = 0, 1
...     while a < n:
...         result.append(a) # result = result + [a], but more efficient
...         a, b = b, a+b
...     return result
...
>>> k=fib2              # save function in variable
>>> f100 = fib2(100)  # save function call
>>> f100
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
>>> # return without an expression argument returns None. Falling off the end of a function also returns None.
```

Default parameters:
```Python
def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'): # This tests whether or not a sequence contains a certain value.
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)
ask_ok('Do you really want to quit?')
ask_ok('OK to overwrite the file?', 2)
ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')

# The default values are evaluated at the point of function definition in the defining scope, so that
i = 5

def f(arg=i):
    print(arg)

i = 6
f() # returns 5

def f(a, L=[]):
    L.append(a)
    return L

print(f(1)) # [1]
print(f(2)) # [1, 2]
print(f(3)) # [1, 2, 3]

# Don't share the default value
def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L
    
print(f(1)) # [1]
print(f(2)) # [2]
print(f(3)) # [3]
```

Keyword argumnts:
```Python
# def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue')

parrot(1000)                                          # 1 positional argument
parrot(voltage=1000)                                  # 1 keyword argument
parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword

# Invalid calls:
parrot()                     # required argument missing
parrot(voltage=5.0, 'dead')  # non-keyword argument after a keyword argument
parrot(110, voltage=220)     # duplicate value for the same argument
parrot(actor='John Cleese')  # unknown keyword argument

# **name contains dictionary with keywords
# *name contains tuple with values
# *name must occur before **name
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])
        
# possible call
cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")
```

Arbitrary Argument Lists:

```Python
# Arbitrary number of arguments. These arguments will be wrapped up in a tuple (see Tuples and Sequences).
# Before the variable number of arguments, zero or more normal arguments may occur.
def write_multiple_items(file, separator, *args):
    file.write(separator.join(args))

# Any formal parameters which occur after the *args parameter are ‘keyword-only’ arguments, meaning that they can only 
# be used as keywords rather than positional arguments.
def concat(*args, sep="/"):
    return sep.join(args)
```

Unpacking argument list:

```Python
>>> list(range(3, 6))            # normal call with separate arguments
[3, 4, 5]
>>> args = [3, 6]
>>> list(range(*args))            # call with arguments unpacked from a list
[3, 4, 5]
```
Lambda expressions:
```Python
# As return
>>> def make_incrementor(n):
...     return lambda x: x + n
...
>>> f = make_incrementor(42)
>>> f(0)
42
>>> f(1)
43

# pass a function as an argument

>>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
>>> pairs.sort(key=lambda pair: pair[1])
>>> pairs
[(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```
Data structures:

```Python
list = []
list.append(1)
list.extend(list)
list.insert(1,'newel')
list.remove(1)
list.pop()
list.clear()
list.index('newe1')
list.count('neww1')
list.sort()
list.reverse()
list.copy()

Smart one-liners
#no loop needed
squares = list(map(lambda x: x**2, range(10))) # option 1
squares = [x**2 for x in range(10)] # option 2


>>> combs = []
>>> for x in [1,2,3]:
...     for y in [3,1,4]:
...         if x != y:
...             combs.append((x, y))

# One-liner for nested loop
combs = [(x,y) for x in [1,2,3] for y in [3,1,4] if x != y]
# [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]

[(x, x**2) for x in range(6)] # list of tuples

# flatten a list
>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]


# Create matrix
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

# transpose rows and columns
>>> [[row[i] for row in matrix] for i in range(4)]
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]

# equivalent to
>>> transposed = []
>>> for i in range(4):
...     transposed.append([row[i] for row in matrix])

# best way to get this job done
>>> list(zip(*matrix))
[(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]

```

Delete elements given their indices:

```Python
>>> a = [-1, 1, 66.25, 333, 333, 1234.5]
>>> del a[0]
>>> a
[1, 66.25, 333, 333, 1234.5]
>>> del a[2:4]
>>> a
[1, 66.25, 1234.5]
>>> del a[:]
>>> a
[]

# delete a variable
del a
```

Singleton tuple:

```Python
>>> empty = ()
>>> singleton = ('hello') # does not work len(lol) = 5
>>> singleton = 'hello',    # <-- note trailing comma
>>> len(empty)
0
>>> len(singleton)
1
>>> singleton
('hello',)
```

Tuple packing:

```Python
t = 12345, 54321, 'hello!'
x, y, z = t     # 3 variables = tuple of 3 elements
```

Sets:
```Python

```

Dictionaries:
```Python
d1 = {'k': 123, 'loool':88}
d1['abc'] = 123
d1 #
del d1['k']

list(d1.keys())   # ['loool', 'abc']
sorted(d1.keys()) # ['abc', 'loool']

# Create dict
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]) # with dict() constructor

{x: x**2 for x in (2, 4, 6)}  # with dict comprehensions

{x: x**2 for x in (2, 4, 6)}  # as keyword argument

# Looping/Iterating through dict

for k, v in d1.items():
    print('key ',k,' val ',v)
# key loool val 88
# key abc val 123

# looping through a sequence with index and val
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
# 0 tic
# 1 tac
# 2 toe

# reversed range
for i in reversed(range(1, 10, 2)):
    print(i)

# sorted returns a new sorted list (source unaltered)
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)
    
# apple (only once 'cause set)
# banana
# orange (only once 'cause set)
# pear

# Change a list while looping over it? 
# No, it is often simpler and safer to create a new list instead

import math
raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
filtered_data = []
for value in raw_data:
    if not math.isnan(value):
        filtered_data.append(value)
        
filtered_data
[56.2, 51.7, 55.3, 52.5, 47.8]

# The Boolean operators and and or are so-called short-circuit operators:
# arguments are evaluated from left to right, evaluation stops as soon as the outcome is determined.
# if A and C are true but B is false, A and B and C does not evaluate the expression C.
# the return value of a short-circuit operator is the last evaluated argument (if no booleans are used)
```

Comparing sequences:
```Python
# Sequence objects may be compared to other objects with the same sequence type. 
# The comparison uses lexicographical ordering
(1, 2, 3)              < (1, 2, 4) # True, two tuples
[1, 2, 3]              < [1, 2, 4] # True, two lists
'ABC' < 'C' < 'Pascal' < 'Python' # True, evaluation from left to right
(1, 2, 3, 4)           < (1, 2, 4) # true, length may differ
(1, 2)                 < (1, 2, -1) # true
(1, 2, 3, 4)           < (1, 2, 3) # false
(1, 2, 3)             == (1.0, 2.0, 3.0) # true, mixed numeric types are compared according to their numeric value
(1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4) # true, stops at 'aa' < 'abc'

# Note that comparing objects of different types with < or > is legal provided that the objects have appropriate comparison
# methods. (for numbers it is their numerical value for instance)
```

___
### Modules
[Docs](https://docs.python.org/3/tutorial/modules.html)

Import a module:
```Python

# this imports the module in file modulfib.py
import modulfib as mf

# call a modules function
mf.fib(100)

# save a modules function locally
fib = mf.fib

# return the modules name
mf.__name__
```

Use a module as a script:
```Python
# this is module fibo.py

# code...

# at the end of the module
if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
    
# now you can run
# python fibo.py <args> to run this code 
# 'cause __name__ is set to __main__
```
Python's module search path:
+ The directory containing the input script (or the current directory when no file is specified).
+ PYTHONPATH (a list of directory names, with the same syntax as the shell variable PATH).
+ The installation-dependent default.

Python saves modules compiled modules in the *__pycache__* directory under the name *module.version.pyc*
Python code is compiles platform-independent (-> Java).

Standard modules:

```Python
# append a folder to PythonPath
import sys
sys.path.append('/ufs/guido/lib/python')

# dir() function
import fibo
dir(fibo)
# ['__name__', 'fib', 'fib2']

# all names in sys
dir(sys)

# all names defined currently
dir()

# dir() does not list the names of built-in functions and variables.
# getting those by using...
import builtins
dir(builtins)

```

[Packages doc](https://docs.python.org/3/tutorial/modules.html#packages)
- Packages are folders containg modules
- can be imported using ```PackageName.ModulName ```

```Python
# this makes the method echofilter from module echo directly available
from sound.effects.echo import echofilter
```
- when using syntax like import item.subitem.subsubitem, each item except for the last must be a package
- the last item can be a module or a package but can’t be a class or function or variable defined in the previous item.

If a package’s __init__.py code defines a list named __all__, it is taken to be the list of module names that should be imported when from package import * is encountered. It is up to the package author to keep this list up-to-date when a new version of the package is released. Package authors may also decide not to support it, if they don’t see a use for importing * from their package. For example, the file sound/effects/__init__.py could contain the following code:

```Python
__all__ = ["echo", "surround", "reverse"]
```
Although certain modules are designed to export only names that follow certain patterns when you use import \*, it is still considered bad practice in production code.

___
### Input and Output

Formatting Output:
```Python
# str() and repr() convert any object into a string
# str()  -> produce human-readable text
# repr() -> meant to generate representations which can be read by the interpreter
>>> s = 'Hello, world.'
>>> str(s)
'Hello, world.'
>>> repr(s)
"'Hello, world.'"
>>> str(1/7)
'0.14285714285714285'
>>> x = 10 * 3.25
>>> y = 200 * 200
>>> s = 'The value of x is ' + repr(x) + ', and y is ' + repr(y) + '...'
>>> print(s)
The value of x is 32.5, and y is 40000...
>>> # The repr() of a string adds string quotes and backslashes:
... hello = 'hello, world\n'
>>> hellos = repr(hello)
>>> print(hellos)
'hello, world\n'
>>> # The argument to repr() may be any Python object:
... repr((x, y, ('spam', 'eggs')))
"(32.5, 40000, ('spam', 'eggs'))"
```

Example (Writing Table of Squares and Cubes):
```Python
>>> for x in range(1, 11):
...     print(repr(x).rjust(2), repr(x*x).rjust(3), end=' ')
...     # Note use of 'end' on previous line
...     print(repr(x*x*x).rjust(4))
...
 1   1    1
 2   4    8
 3   9   27
 4  16   64
 5  25  125
 6  36  216
 7  49  343
 8  64  512
 9  81  729
10 100 1000

# The following generates the same output
>>> for x in range(1, 11):
...     print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

str.rjust() # right-justifies
str.ljust() # left-justifies
str.center() # centers
```

Usage of *format()*:
```Python
print('We are the {} who say "{}!"'.format('knights', 'Ni'))
# We are the knights who say "Ni!"

# Brackets with values (format fields)
print('{0} and {1}'.format('spam', 'eggs'))
# spam and eggs
print('{1} and {0}'.format('spam', 'eggs'))
# eggs and spam

# with keyword arguments
print('This {food} is {adjective}.'.format(food='spam', adjective='absolutely horrible'))
# This spam is absolutely horrible.

# combining both
print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred',
                                                       other='Georg'))
# The story of Bill, Manfred, and Georg.

# '!a' (apply ascii()), '!s' (apply str()) and '!r' (apply repr()) can be used to convert the value before it is formatted
contents = 'eels'
print('My hovercraft is full of {}.'.format(contents))
# My hovercraft is full of eels.
print('My hovercraft is full of {!r}.'.format(contents))
# My hovercraft is full of 'eels'.

# Use : to add additional format codes
import math
print('The value of PI is approximately {0:.3f}.'.format(math.pi))
# The value of PI is approximately 3.142.

# Use integer after the ':' for minimum character width
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print('{0:10} ==> {1:10d}'.format(name, phone))
# Jack       ==>       4098
# Dcab       ==>       7678
# Sjoerd     ==>       4127


# Referencing by name instead of position
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {0[Jack]:d}; Sjoerd: {0[Sjoerd]:d}; '
      'Dcab: {0[Dcab]:d}'.format(table))
# Jack: 4098; Sjoerd: 4127; Dcab: 8637678

# Old-school formatting
print('The value of PI is approximately %5.3f.' % math.pi)
# The value of PI is approximately 3.142.
```
___
### Files

Reading and Writing Files:
```Python
# open(filename, mode) mode='r', 'w' or 'r+' ('r' if omitted)
# append 'b' for binary mode (default is text mode)
f = open('workfile', 'w')

# Good practice: using with (ensures closing, even if exception occurs)
with open('workfile') as f:
     read_data = f.read()
f.closed
# True

# Without 'with' one must close the file explicitly
f.close()
```

Methods of File Objects:
```Python
# Read entire file
f.read()

# read until newline \n
f.readline()

# looping over lines of a file
for line in f:
    print(line, end='')
# This is the first line of the file.
# Second line of the file

# read lines of a file into a list
list(f)
f.readlines()

# Writing
# f.write(string)
f.write('This is a test\n')
# 15  (Number of chars written)

# Other types must be converted first (either to string or to a bytes object in bin mode)
value = ('the answer', 42)
s = str(value)  # convert the tuple to string
f.write(s)

# get the current position of the object (returns number of bytes from the beginning when in byte mode)
f.tell()

# change the file object's position with f.seek(offset, from_what)
# from_what=0 (beginning of file, default)
# from_what=1 (current file position)
# from_what=2 (end of file)

f = open('workfile', 'rb+')
f.write(b'0123456789abcdef')
# 16
f.seek(5)      # Go to the 6th byte in the file
# 5
f.read(1)
# b'5'
f.seek(-3, 2)  # Go to the 3rd byte before the end
# 13
f.read(1)
# b'd'

# In text files (those opened without a b in the mode string), only seeks relative to the beginning of the file are allowed
# Other values produce undefined behaviour
```

Save structured data with ```json```:
[doc stelle json](https://docs.python.org/3/tutorial/inputoutput.html#saving-structured-data-with-json)
```Python
# create json representation from python object
import json
json.dumps([1, 'simple', 'list'])
# '[1, "simple", "list"]'

# dump() serializes into a text file,
json.dump(x, f) # x = object, f = opened textfile for writing

# for decoding
x = json.load(f) # f = opened textfile for reading
# This simple serialization technique can handle lists and dictionaries
# but serializing arbitrary class instances in JSON requires a bit of extra effort
```
See (pickle doc)[https://docs.python.org/3/library/pickle.html#module-pickle] for information considering the transformation of arbitrary python objects into json representation

___
### Exception handling

[docs exception](https://docs.python.org/3/tutorial/errors.html)

Exception throwing:
```Python
for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except OSError:
        print('cannot open', arg)
    else:       # if the try clause does not raise an exception
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()
```

User-defined Exceptions:
```Python
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class TransitionError(Error):
    """Raised when an operation attempts a state transition that's not
    allowed.

    Attributes:
        previous -- state at beginning of transition
        next -- attempted new state
        message -- explanation of why the specific transition is not allowed
    """

    def __init__(self, previous, next, message):
        self.previous = previous
        self.next = next
        self.message = message
```

Clean-up actions:
```Python
def divide(x, y):
     try:
         result = x / y
     except ZeroDivisionError:
         print "division by zero!"
     else:
         print "result is", result
     finally:
         print "executing finally clause"
```

# Pycharm tricks

+ Where to change key bindings:
```
File | Settings | Keymap
```

+ *Autoformat *in Eclipse (CTRL + Shift + F) the action is called *Reformat code* in PyCharm. Located in the Code menu, somewhere in the middle.

+ Where to activate documentation-popup on mouse-hover
```
File | Settings | Editor | Show quick doc on mouse move
```


