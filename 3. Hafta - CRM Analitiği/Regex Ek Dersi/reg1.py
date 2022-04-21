# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility

import re

regex = r".*(?<=user\s)(\w+)\sfrom\s([\d\.]+)"

test_str = ("May 21 19:44:14 lisans sshd[20935]: Invalid user qbase from 5.39.6.89\n"
            "May 21 19:44:14 lisans sshd[20935]: input_userauth_request: invalid user qbase [preauth]\n"
            "May 21 19:44:14 lisans sshd[20935]: pam_unix(sshd:auth): check pass; user unknown\n"
            "May 21 19:44:14 lisans sshd[20935]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=5.39.6.89\n"
            "May 21 19:44:16 lisans sshd[20935]: Failed password for invalid user qbase from 5.39.6.89 port 49718 ssh2\n"
            "May 21 19:44:16 lisans sshd[20935]: Received disconnect from 5.39.6.89 port 49718:11: Normal Shutdown, Thank you for playing [preauth]\n"
            "May 21 19:44:16 lisans sshd[20935]: Disconnected from 5.39.6.89 port 49718 [preauth]\n"
            "May 21 19:44:27 lisans sshd[20937]: Invalid user admin from 193.32.163.89\n"
            "May 21 19:44:27 lisans sshd[20937]: input_userauth_request: invalid user admin [preauth]\n"
            "May 21 19:44:27 lisans sshd[20937]: pam_unix(sshd:auth): check pass; user unknown\n"
            "May 21 19:44:27 lisans sshd[20937]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=193.32.163.89\n"
            "May 21 19:44:29 lisans sshd[20937]: Failed password for invalid user admin from 193.32.163.89 port 54360 ssh2\n"
            "May 21 19:44:29 lisans sshd[20937]: Disconnecting: Change of username or service not allowed: (admin,ssh-connection) -> (user,ssh-connection) [preauth]\n"
            "May 21 19:44:52 lisans sshd[20939]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=112.85.42.232  user=root\n"
            "May 21 19:44:54 lisans sshd[20939]: Failed password for root from 112.85.42.232 port 10613 ssh2\n"
            "May 21 19:44:59 lisans sshd[20939]: message repeated 2 times: [ Failed password for root from 112.85.42.232 port 10613 ssh2]\n"
            "May 21 19:44:59 lisans sshd[20939]: Received disconnect from 112.85.42.232 port 10613:11:  [preauth]\n"
            "May 21 19:44:59 lisans sshd[20939]: Disconnected from 112.85.42.232 port 10613 [preauth]\n"
            "May 21 19:44:59 lisans sshd[20939]: PAM 2 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=112.85.42.232  user=root\n"
            "May 21 19:45:01 lisans CRON[20941]: pam_unix(cron:session): session opened for user root by (uid=0)\n"
            "May 21 19:45:01 lisans CRON[20941]: pam_unix(cron:session): session closed for user root\n"
            "May 21 19:45:28 lisans sshd[20944]: Invalid user ev from 92.154.53.93\n"
            "May 21 19:45:28 lisans sshd[20944]: input_userauth_request: invalid user ev [preauth]")

matches = re.finditer(regex, test_str)

for matchNum, match in enumerate(matches, start=1):

    print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(), end=match.end(), match=match.group()))

    for groupNum in range(0, len(match.groups())):
        groupNum = groupNum + 1

        print("Group {groupNum} found at {start}-{end}: {group}".format(groupNum=groupNum, start=match.start(groupNum), end=match.end(groupNum),
                                                                        group=match.group(groupNum)))

# Note: for Python 2.7 compatibility, use ur"" to prefix the regex and u"" to prefix the test string and substitution.
