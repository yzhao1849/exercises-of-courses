def if_palindrome(s):
    newstring=[]
    for i in s:
        if i.isalpha():
            newstring.append(i.lower())
    #print(newstring)
    j=0
    while(newstring[j]==newstring[len(newstring)-1-j] and j<=int(len(newstring)/2)):
        j=j+1
    if j>int(len(newstring)/2):
        return True
    else:
        return False

s1="Mr. Owl ate my metal worm"
print(if_palindrome(s1))

s2="Dammit I'm Mad"
print(if_palindrome(s2))

s3="Rats live on no evil star"
print(if_palindrome(s3))

s4="Rats live on evil star"
print(if_palindrome(s4))

s5="Mice live on no evil star"
print(if_palindrome(s5))

