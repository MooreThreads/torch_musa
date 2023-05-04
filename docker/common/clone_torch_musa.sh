#!/usr/bin/expect -f
set username [lindex $argv 0]
set password [lindex $argv 1]
set repository "https://github.mthreads.com/mthreads/torch_musa.git"

spawn git clone $repository
expect "Username for 'https://github.mthreads.com':"
send "$username\r"
expect "Password for 'https://$username@github.mthreads.com':"
send "$password\r"

expect eof
