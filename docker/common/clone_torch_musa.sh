#!/usr/bin/expect -f
set username [lindex $argv 0]
set password [lindex $argv 1]
set repository "https://sh-code.mthreads.com/ai/torch_musa.git"

spawn git clone $repository
expect "Username for 'https://sh-code.mthreads.com':"
send "$username\r"
expect "Password for 'https://$username@sh-code.mthreads.com':"
send "$password\r"

expect eof
