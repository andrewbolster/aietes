# source http://projects.puppetlabs.com/projects/1/wiki/Apt_Keys_Patterns
define apt::key($ensure = present, $keyserver = "keyserver.ubuntu.com") {
  $grep_for_key = "apt-key list | grep '^pub' | sed -r 's.^pub\\s+\\w+/..' | grep '^$name'"
  case $ensure {
    present: {
      exec { "Import $name to apt keystore":
        path        => "/bin:/usr/bin",
        environment => "HOME=/root",
        command     => "gpg --keyserver $keyserver --recv-keys $name && gpg --export --armor $name | sudo apt-key add -",
        user        => "root",
        group       => "root",
        unless      => "$grep_for_key",
        logoutput   => true,
      }
    }
    absent:  {
      exec { "Remove $name from apt keystore":
        path    => "/bin:/usr/bin",
        environment => "HOME=/root",
        command => "apt-key del $name",
        user    => "root",
        group   => "root",
        onlyif  => "$grep_for_key",
      }
    }
    default: {
      fail "Invalid 'ensure' value '$ensure' for apt::key"
    }
  }
}

class kivy-teamppa {
  file { '/etc/apt/sources.list.d/kivy.list':
    content => "deb http://ppa.launchpad.net/kivy-team/kivy/ubuntu precise main
deb-src http://ppa.launchpad.net/kivy-team/kivy/ubuntu precise main"
  }
  apt::key{ "A863D2D6": }
  package { "kivy": ensure => latest, require=> Exec['apt-get update']}
}

group { "puppet": ensure => "present"}

package { "build-essential": ensure => latest}
package { "python-numpy": ensure => latest}
package { "python-scipy": ensure => latest}
package { "python-matplotlib": ensure => latest}
package { "python-simpy": ensure => latest}
package { "python-pydot": ensure => latest}
package { "python-wxgtk2.8": ensure => latest}
package { "libfreetype6-dev": ensure => latest}
package { "mencoder": ensure => latest}


File { owner => 0, group => 0, mode => 0644 }
file { '/etc/motd':
   content => "Welcome to your Vagrant-built virtual machine!
               Managed by Puppet.\n"
}
