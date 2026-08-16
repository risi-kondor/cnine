#ifndef _POSIXsubprocess
#define _POSIXsubprocess

#include <spawn.h>  
#include <sys/wait.h>
#include <unistd.h>  
#include <chrono>
#include <csignal>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "Cnine_base.hpp"


extern char** environ;

namespace cnine{

  enum class WaitResult{exited,timed_out};


  class POSIXsubprocess{
  public:

    pid_t pid;
    int status_out;


    POSIXsubprocess(string command, vector<string> _args, const double timeout=-1){
      CNINE_ASSRT(_args.size()<=8);
      char* args[10];
      auto arg0p=std::make_unique<char[]>(command.size()+1);
      strcpy(arg0p.get(),command.c_str());
      args[0]=arg0p.get();

      vector<unique_ptr<char[]> > argp(_args.size());
      for(int i=0; i<_args.size(); i++){
	argp[i]=std::make_unique<char[]>(_args[i].size()+1);
	strcpy(argp[i].get(),_args[i].c_str());
	args[i+1]=argp[i].get();
      }
      args[_args.size()+1]=nullptr;

      int rc=posix_spawnp(&pid,command.c_str(),nullptr,nullptr,args,environ);

      if (rc!=0) throw std::runtime_error(std::string("posix_spawn failed: ") + std::strerror(rc));

      if(timeout<=0){
	int status=0;
	if(waitpid(pid, &status, 0)==-1)
	  throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
	if(!WIFEXITED(status) || WEXITSTATUS(status) != 0)
	  throw std::runtime_error("autotuning worker failed");
      }else{
	wait_with_timeout(timeout);
      }

    }


    WaitResult wait_with_timeout(const double timeout){
      using clock=std::chrono::steady_clock;
      const auto deadline=clock::now()+std::chrono::duration<double>(timeout);

      for(;;){
	pid_t r=waitpid(pid,&status_out,WNOHANG);
	if(r==pid) return WaitResult::exited;
	if(r==-1){
	  if(errno==EINTR) continue;
	  throw std::system_error(errno, std::generic_category(),"waitpid");
	}
	if(clock::now()>=deadline)break;
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    
      cout<<"TIMEOUT!"<<endl;

      // Request clean shutdown first.
      if(kill(pid,SIGTERM)==-1 && errno!=ESRCH)
	throw std::system_error(errno, std::generic_category(), "kill(SIGTERM)");
    
      // Allow, say, 100 ms for handlers / normal shutdown.
      const auto grace_deadline=clock::now()+std::chrono::milliseconds(100);

      for(;;){
	pid_t r=waitpid(pid,&status_out,WNOHANG);
	if(r==pid) return WaitResult::timed_out;

	if(r==-1){
	  if(errno==EINTR) continue;
	  if (errno==ECHILD) return WaitResult::timed_out;
	  throw std::system_error(errno, std::generic_category(), "waitpid");
	}
	if(clock::now()>=grace_deadline) break;
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      if(kill(pid, SIGKILL)==-1 && errno!=ESRCH)
	throw std::system_error(errno, std::generic_category(), "kill(SIGKILL)");

      // Mandatory: collect termination status and prevent a zombie.
      while(waitpid(pid, &status_out, 0)==-1){
	if(errno!=EINTR){
	  if(errno==ECHILD) break;
	  throw std::system_error(errno, std::generic_category(), "waitpid");
	}
      }

      return WaitResult::timed_out;
    }

  };

}

#endif 
