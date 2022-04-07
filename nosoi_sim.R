## Initialize stable working environment and store time of intiation
rm(list=ls())

#setwd("/Users/macbook/Dropbox (UFL)/DYNAMITE/HIVdynamite/nosoi_simulations")
# List of packages for session
.packages <-  c("phytools", "ape", "parallel", "ggplot2", "viridis", "igraph", "ggnetwork", 
                "ggpubr", "ggtree", "treeio", "ape", "remotes", "dplyr", "plyr", "phyclust") 
github_packages <- c("slequime/nosoi", "emillykkejensen/familyR") 

# Install CRAN packages (if not already installed)
.inst <- .packages %in% installed.packages()
if(length(.packages[!.inst]) > 0) install.packages(.packages[!.inst])
.inst_github <- .packages %in% installed.packages()
## Install GitHub packages(if not already installed)
if(length(github_packages[!.inst_github]) > 0) try(remotes::install_github(github_packages[!.inst_github]))
if(length(github_packages[!.inst_github]) > 0) try(devtools::install_github(github_packages[!.inst_github]))

# Load packages into session 
lapply(.packages, require, character.only=TRUE)
lapply(gsub(".+\\/(.+)", "\\1", github_packages), require, character.only=TRUE)

### Set seeds #############################################################################
# get simulation parameters

args = commandArgs(trailingOnly=TRUE)
sim_index = as.numeric(args[1]) # arg 1 fraction new
#sim_index="test"
seeds = readLines('seeds.txt')
set.seed(seeds[sim_index])
numCores = detectCores()


## Matrix generation #######################################################################
traits <- data.frame(location=rbind('A', 'B', 'C', 'D', 'E', 'F.1', 'F.2', 'G.1', 'G.2'))
Q <-list()
for (column in 1:ncol(traits)) {
  suppressWarnings({ ## We know it fills diagonals with NAs
    Q[[column]] <- diag(unique(traits[,column]), nrow = length(unique(traits[,column])))
  })
  #    diag(Q[[column]]) = 1-nrow(Q[[column]])
  diag(Q[[column]]) = 0
  non.diag <- 1/(nrow(Q[[column]])-1)
  Q[[column]][lower.tri(Q[[column]])] <- non.diag
  Q[[column]][upper.tri(Q[[column]])] <- non.diag
  colnames(Q[[column]]) <- rownames(Q[[column]]) <- unique(traits[,column])
  ## Modify Q matrix to allow for birth of cluster I from cluster H
#  Q[[column]][nrow(traits)-1,] <- c(rep(0.0, nrow(traits)-1),1.0) # H only gives rise to I
#  Q[[column]][1,] <- c(0.0,rep(1/(nrow(traits)-2),nrow(traits)-2),0.0) # A cannot give rise to I; the remaining clusters can stay at 0.125 because they have a 0 probability of leaving (see below)
  }


Q <- plyr::compact(Q)
names(Q) <- colnames(traits[-1])
Q


# OR
#create matrix layout
# df.matrix = df %>% spread(To, value=N, fill= 0)
# df.matrix = as.matrix(df.matrix[-1])
# rownames(df.matrix) = colnames(df.matrix)
# 
# #Get probabilities (beware, rows should sum up to 1)
# df_transpose = t(df.matrix)
# probabilities <- apply(df_transpose, 1, function(i) i/sum(i))
# transition.matrix = t(probabilities)



# #pExit daily probability for a host to leave the simulation (either cured, died, etc.).
p_Exit_fct  <- function(t, t_sero){ 
  if(t <= t_sero){p=0}
  if(t > t_sero){p=0.95} 
  return(p)
}
t_sero_fct <- function(x){rnorm(x,14,2)} #Approximately 14 days

#pMove probability (per unit of time) for a host to do move, i.e. to leave its current state (for example, leaving state “A”). 
#It should not be confused with the probabilities extracted from the structure.matrix, which represent the probability to go 
#to a specific location once a movement is ongoing (for example, going to “B” or “C” while coming from “A”).
p_Move_fct  <- function(t, current.in, host.count){
  
  if(current.in=="A" & host.count < 2){return(0)}
  if(current.in=="A" & host.count >= 2){return(0.00075)} # Wait a couple of weeks before initiation of clusters
  if(current.in=="B"){return(0)}
  if(current.in=="C"){return(0)}
  if(current.in=="D"){return(0)}
  if(current.in=="E"){return(0)}
  if(current.in=="F.1"){return(0)}
  if(current.in=="F.2"){return(0)}
  if(current.in=="G.1"){return(0)}
  if(current.in=="G.2"){return(0)}
}
#  t_clust_fct <- function(x){rnorm(x,mean = 3,sd=1)}




n_contact_fct = function(t, current.in, host.count){
  
  
  starting_n0_low <- round(rnorm(100, 4, 1))
  n0_low <- sample(starting_n0_low, 1)
  assign("n0_low", n0_low, envir = globalenv())
  
  growth <- list()
  growth$exp_growth <- abs(round(n0_low*10*exp(0.0005*host.count)/((10-n0_low)+n0_low*exp(0.0005*host.count))))
  growth$lin_growth <- abs(round(0.025*host.count+1))
  
  starting_n0_hi <- round(rnorm(100, 6, 1))
  n0_hi <- sample(starting_n0_hi, 1)
  assign("n0_hi", n0_hi, envir = globalenv())
  
  decay <- list()
  decay$exp_decay <- abs(round(n0_hi*10*exp(-0.05*host.count)/((10-n0_hi)+n0_hi*exp(-0.05*host.count))))
  decay$lin_decay <- abs(round(-0.025*host.count+(n0_hi*4)))
  
    
  if(current.in=="A") {p=abs(round(rnorm(1, 16, 1), 0))}
  if(current.in=="B") {p=abs(round(rnorm(1, 4, 1), 0))}
  if(current.in=="C") {p=abs(round(rnorm(1, 4, 1), 0))}
  if(current.in=="D") {p=abs(round(rnorm(1, 6, 1), 0))}
  if(current.in=="E") {p=abs(round(rnorm(1, 4, 1), 0))}
  if(current.in=="F.1") {p=growth[[1]]}
  if(current.in=="F.2") {p=growth[[2]]}
  if(current.in=="G.1") {p=decay[[1]]}
  if(current.in=="G.2") {p=decay[[2]]}
  return(p)
}

#pTrans represents the probability of transmission over time (when a contact occurs).
# in the form of a threshold function: before a certain amount of time since initial infection, the host does not transmit (incubation time, which we call t_incub), and after that time, it will transmit with a certain (constant) probability (which we call p_max). This function is dependent of the time since the host’s infection t.
p_Trans_fct <- function(t, current.in, host.count, t_incub){
  if(t < t_incub){p=0}
  if(t >= t_incub & current.in=="A"){p=0.015}
  if(t >= t_incub & current.in=="B"){p=0.1}
  if(t >= t_incub & current.in=="C"){p=0.15}
  if(t >= t_incub & current.in=="D"){p=0.1}
  if(t >= t_incub & current.in=="E"){p=0.2}
  if(t >= t_incub & current.in=="F.1"){p=0.15}
  if(t >= t_incub & current.in=="F.2"){p=0.15}
  if(t >= t_incub & current.in=="G.1"){p=0.15}
  if(t >= t_incub & current.in=="G.2"){p=0.15}
  return(p)
}

t_incub_fct <- function(x){rnorm(x,mean = 5,sd=2)} #Approximately 4.2 days
#p_max_fct <- function(x){rbeta(x,shape1 = 1,shape2=8)}# Mean of roughly 0.10

# Starting the simulation ------------------------------------

#set.seed(111)

SimulationSingle <- nosoiSim(type="single", # Number of hosts
                             popStructure="discrete", #discrete or continuous
                             structure.matrix = Q[[1]], # prob matrix defined above (row sums to 1, with diags as zero)
                             length.sim = 365, # Max number of time units (can be days, months, weeks, etc.)
                             max.infected = 10000, #maximum number of individuals that can be infected during the simulation.
                             init.individuals = 1, #number of individuals (an integer above 1) that will start a transmission chain. Keep in mind that you will have as many transmission chains as initial individuals, which is equivalent as launching a number of independent nosoi simulations.
                             init.structure = "A",
                             
                             pExit = p_Exit_fct,
                             param.pExit=list(t_sero=t_sero_fct),
                             timeDep.pExit=FALSE,
                             diff.pExit=FALSE,
                             
                             pMove = p_Move_fct,
                             hostCount.pMove=TRUE,
                             param.pMove=NA,
                             timeDep.pMove=FALSE,
                             diff.pMove=TRUE,
                             
                             nContact=n_contact_fct,
                             hostCount.nContact=TRUE,
                             param.nContact = NA,
                             timeDep.nContact=FALSE,
                             diff.nContact=TRUE,
                             
                             pTrans = p_Trans_fct,
                             hostCount.pTrans=TRUE,
                             param.pTrans = list(t_incub=t_incub_fct),
                             timeDep.pTrans=FALSE,
                             diff.pTrans=TRUE,
                             
                             prefix.host="S",
                             print.progress=TRUE,
                             print.step=100)

sum_sim <- summary(SimulationSingle)
cumulative.table <- getCumulative(SimulationSingle)
dynamics.table <- getDynamic(SimulationSingle)
cum.p <- ggplot(data=cumulative.table, aes(x=t, y=Count)) + geom_line() + theme_minimal() +
  labs(x="Time (t)",y="Cumulative count of infected hosts") + scale_y_log10()
cum.p.c <- ggplot(data=dynamics.table, aes(x=t, y=Count, color=state)) + geom_line() + theme_minimal() +
  labs(x="Time (t)",y="Number of active infected hosts") + scale_y_log10()


# ggpubr::ggarrange(cum.p, cum.p.c, widths = 2, heights = 1, legend="right")
# ggsave("simtest_max10000.png", plot=last_plot())


## Grab tree #########################################################################################
save.data <- function(){
  sim.tree <- getTransmissionTree(SimulationSingle)
  # ggtree(test.nosoiA.tree, color = "gray30") + geom_nodepoint(aes(color=state)) + geom_tippoint(aes(color=state)) + 
  #   theme_tree2() + xlab("Time (t)") + theme(legend.position = c(0,0.8), 
  #                                            legend.title = element_blank(),
  #                                            legend.key = element_blank()) 
  #set.seed(5905950) 
  
  # Get sampled tree
  getSample <- function(SimulationSingle) {
    int.nodes <- sample((Ntip(sim.tree@phylo)+2):(Ntip(sim.tree@phylo)+sim.tree@phylo$Nnode)) #randomize order of internal nodes
    n <- unique(sum_sim$dynamics$state[sum_sim$dynamics$state != "A"]) 
    s <- seq(5,100,1)

    state.list <- list()
    for (i in 1:length(n)) {
      nodes <- sample(sim.tree@data$node[sim.tree@data$state==n[i]])
      state <- n[i]
      state.list[[i]] <- data.frame(nodes=nodes, state=state)
      state.list[[i]] <- dplyr::filter(state.list[[i]],
                                       nodes %in% int.nodes)
    } #End state.list generation
      

    sampleState <- function(state) {
      max_length <- length(state$nodes)
      tcs <- data.frame(taxa=NA, state=NA)
      i=1
      while (isTRUE(is.na(tcs$taxa) & i <= max_length)) {
        n=state$nodes[i]    
        tcs.phylo <- extract.clade(sim.tree@phylo, n)
        tcs.taxa <- subset(sim.tree@data, sim.tree@data$host %in% tcs.phylo$tip.label)
        unique_hosts <- unique(tcs.taxa$host)
        
        lo <- 6*last(cum.p$data$t)/10000 ; hi <- 1
        r <- 0.01
        C <- exp(-r*hi); D <- exp(-r*lo)
        n <- 100
        U <- runif(n,min=C,max=D)
        X <- (1/r)*log(1/U)
        #hist(X,breaks=10,xlim=c(0,1))
        sf <- sample(X, 1)
        
        sampled_hosts <- sample(unique_hosts, round(sf*length(unique_hosts)), replace=F)
        
        if(isTRUE(length(grep(state$state[1], tcs.taxa$state)) >= 0.95*length(tcs.taxa$state) &
           length(sampled_hosts) %in% s)) {
          tcs <- data.frame(taxa=sampled_hosts, state=state$state[1], sampling_fraction=sf, cluster_size=length(unique_hosts)) 
           i=i
        } else {
          tcs <- data.frame(taxa=NA, state=NA, sampling_fraction=NA, cluster_size=NA)
          i=i+1}
      } # End while loop
       return(tcs)
    } # End function
   
   
    tcs.list <- mclapply(state.list, sampleState, mc.cores=numCores)
    s_rand <- round(rnorm(10,50,10))
    group_A <- sample(sim.tree@data$node[sim.tree@data$state=="A"])
    group_A <- group_A[group_A %in% int.nodes]
    
    sampleA <- function(group_A) {
      cluster_A <- NULL
      while(is.null(cluster_A)){
        true.cluster.phylo <- extract.clade(sim.tree@phylo, sample(group_A, 1))
        true.cluster.taxa <- unique(subset(sim.tree@data, sim.tree@data$host %in% true.cluster.phylo$tip.label))
        unique_hosts <- unique(true.cluster.taxa$host)
 
        lo <- 6*last(cum.p$data$t)/10000 ; hi <- 1
        r <- 0.01
        C <- exp(-r*hi); D <- exp(-r*lo)
        n <- 100
        U <- runif(n,min=C,max=D)
        X <- (1/r)*log(1/U)
        #hist(X,breaks=10,xlim=c(0,1))
        sf <- sample(X, 1)
        
        sampled_hosts <- sample(unique_hosts, round(sf*length(unique_hosts)), replace=F)
        
        if(isTRUE(length(sampled_hosts) %in% s_rand &
           length(sampled_hosts) >= 3 &
           length(grep("A", true.cluster.taxa$state)) >= 0.95*length(true.cluster.taxa$state))) {
          cluster_A <- data.frame(taxa=sampled_hosts, state="A", sampling_fraction=sf, cluster_size=length(unique_hosts)) 
        }
      }
       return(cluster_A)
    }
    
    group_A_cluster <- sampleA(group_A)
    tcs.list <- append(list(group_A_cluster), tcs.list) %>%
      Filter(function(a) any(!is.na(a)), .)
 
     
    table.hosts <- getTableHosts(SimulationSingle, pop="A")
    sampled.hosts <- sample(table.hosts$hosts.ID[table.hosts$current.in=="A"], round(6*last(cum.p$data$t)), replace=F)
    ## Add these individuals to list of randomly sampled individuals
    sampled.hosts <- unique(c(sampled.hosts, unname(unlist(lapply(tcs.list, "[", 'taxa')))))
    ## Extract tree for list of individuals from the full simulation tree
    sampled.tree <- sampleTransmissionTreeFromExiting(sim.tree, sampled.hosts)

    #sampled.tree <- keep.tip(sim.tree@phylo, sampled.hosts)
    
 
    
    list_nodes <- mclapply(tcs.list, function(x) {
      n <- findMRCA(sampled.tree@phylo, x$taxa)
      x2 <- as_tibble(extract.clade(sampled.tree@phylo, n))
      return(x2)
    }, mc.cores=numCores)
    
    # Give names to clusters (c1..n)
    for (i in seq_along(list_nodes)) {
      names(list_nodes)[i] <- paste0("c", i)
      names(tcs.list)[i] <- paste0("c", i)
    }
    
    assign("tcs.list", tcs.list, envir = globalenv()) # Remember now a tibble
    assign("list_nodes", list_nodes, envir = globalenv()) # Remember now a tibble
    
    
    return(as_tibble(sampled.tree))
  }
  sampled.tree <- getSample(SimulationSingle)
  
 
  trans_clusters <- lapply(tcs.list, function(x) {
    x <- subset(sampled.tree, sampled.tree$label %in% x$taxa)
    x$date <- as.Date(x$time, origin="2019-12-31")
    x$label <- paste(x$label, x$state, x$date, sep="|")
    return(x)
  })
  
  saveRDS(trans_clusters, paste0("trans_clusters_", sim_index, ".rds"))
  
  
  # Transform sampled tree into a tbl object and assign cluster IDs to internal and external nodes found in list of clusters
  sampled.tree$cluster_id <- "Background"
  for (i in seq_along(list_nodes)) {
    for (j in 1:length(sampled.tree$label)) {
      if (isTRUE(sampled.tree$label[j] %in% list_nodes[[i]]$label)) {
        sampled.tree$cluster_id[j] = names(list_nodes)[i]
      }
    }
  }
  

  sampled.tree$date <- as.Date(sampled.tree$time, origin="2019-12-31")
  sampled.tree$label <- paste(sampled.tree$label, sampled.tree$state, sampled.tree$date, sep="|")
  sampled.tree <- dplyr::select(sampled.tree, parent, node, branch.length, label, cluster_id) %>%
    as_tibble()
  class(sampled.tree) = c("tbl_tree", class(sampled.tree))
  
  t2 <- as.treedata(sampled.tree)
  write.beast(t2, paste0('sim_', sim_index, "_sampled_10000.tree")) #### NEED THIS OUTPUT####################################

  
  
  ##### Sequence evolution along tree
  text<-write.tree(t2@phylo)
  strip.nodelabels<-function(text){
    obj<-strsplit(text,"")[[1]]
    cp<-grep(")",obj)
    csc<-c(grep(":",obj),length(obj))
    exc<-cbind(cp,sapply(cp,function(x,y) y[which(y>x)[1]],y=csc))
    exc<-exc[(exc[,2]-exc[,1])>1,]
    inc<-rep(TRUE,length(obj))
    if(nrow(exc)>0) for(i in 1:nrow(exc))
      inc[(exc[i,1]+1):(exc[i,2]-1)]<-FALSE
    paste(obj[inc],collapse="")
  }
  t2_phylo <- strip.nodelabels(text)
  t2_phylo <- read.tree(text=t2_phylo)
  t2@phylo <- t2_phylo
  
  text <- write.tree(t2@phylo)
  text <- strip.nodelabels(text)
  text <- read.tree(text=text)
  text <- multi2di(text)
  seqdata <- seqgen(opts="-s8.219178e-04 -mGTR -i0.601 -a2.35 -r0.32512,1.07402,0.26711,0.25277,2.89976,1.00000 -f0.299,0.183,0.196,0.322", rooted.tree=text)
  seqdata <- as.vector(seqdata)
  
  write.table(seqdata, paste0("seqdata_", sim_index, ".phy"), 
              quote=F, row.names = F, col.names = F)
  
  ## Export mean R0 for entire simulation for benchmarking
  
  
  true.cluster.dyn <- function(conf.level=0.95){
    cluster_dynamics <- data.frame(sim=sim_index,
                                   state=rbind('A', 'B', 'C', 'D', 'E', 'F.1', 'F.2', 'G.1', 'G.2'),
                                   dynamic = rbind('static', 'static', 'static', 'static', 'static', 
                                                   'growth', 'growth', 'decay', 'decay'),
                                   dynamic_model = rbind(NA, NA, NA, NA, NA, 
                                                         'exp', 'lin', 'exp', 'lin')
    )
    
    cluster_dynamics$cluster_id <- NA
    cluster_dynamics$v_sampling_fraction <- NA
    cluster_dynamics$v_cluster_size <- NA
    for (i in seq_along(tcs.list)) {
      for (j in 1:nrow(cluster_dynamics)) {
        if(isTRUE(cluster_dynamics$state[j] == names(which.max(table(tcs.list[[i]]$state))))) {
          cluster_dynamics$cluster_id[j] <- names(tcs.list)[i]
          cluster_dynamics$v_sampling_fraction[j] <- tcs.list[[i]]$sampling_fraction[1]
          cluster_dynamics$v_cluster_size[j] <- tcs.list[[i]]$cluster_size[1]
        } 
      }
    }
    cluster_dynamics <- cluster_dynamics %>%
      filter(!is.na(cluster_id))
       
    
    ## Estimate R0 for each cluster based on paramaters used in simulation
    # s <- 100 # sample size of 100
    # n_contacts_BC <- rnorm(s, 4, 1)
    # n_contacts_D <- rnorm(s, 6, 1)
    # n_contacts_E <- rnorm(s, 4, 1)
    # n_contacts_F <- n0_low
    # n_contacts_G <- n0_hi
    # 
    # t_incub <- rnorm(s, 5, 2)
    # t_exit <- rnorm(s, 14, 2)
    # p_trans_BD <- 0.1
    # p_trans_CFG <- 0.15
    # p_trans_E <- 0.2
    # 
    # 
    # Z=qnorm(0.5*(1 + conf.level))
    # 
    # R0_B = sample(n_contacts_BC*p_trans_BD*(t_exit-t_incub),
    #               size=s, replace=F)
    # R0_C = sample(n_contacts_BC*p_trans_CFG*(t_exit-t_incub),
    #               size=s, replace=F)
    # R0_D = sample(n_contacts_D*p_trans_BD*(t_exit-t_incub),
    #               size=s, replace=F)
    # R0_E = sample(n_contacts_E*p_trans_E*(t_exit-t_incub),
    #               size=s, replace=F)
    # R0_F = sample(n_contacts_F*p_trans_CFG*(t_exit-t_incub),
    #               size=s, replace=F)
    # R0_G = sample(n_contacts_G*p_trans_CFG*(t_exit-t_incub),
    #               size=s, replace=F)
    # 
    # logR0_B = log(R0_B)
    # logR0_C = log(R0_C)
    # logR0_D = log(R0_D)
    # logR0_E = log(R0_E)
    # logR0_F = log(R0_F)
    # logR0_G = log(R0_G)
    # 
    # SElogR0_B = sd(logR0_B, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # SElogR0_C = sd(logR0_C, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # SElogR0_D = sd(logR0_D, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # SElogR0_E = sd(logR0_E, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # SElogR0_F = sd(logR0_F, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # SElogR0_G = sd(logR0_G, na.rm=T)/sqrt(s) # standard deviation of 2, sample size of 10
    # 
    # lower_B = exp(mean(logR0_B, na.rm=T) - Z*SElogR0_B) 
    # upper_B = exp(mean(logR0_B, na.rm=T) + Z*SElogR0_B) 
    # lower_C = exp(mean(logR0_C, na.rm=T) - Z*SElogR0_C) 
    # upper_C = exp(mean(logR0_C, na.rm=T) + Z*SElogR0_C) 
    # lower_D = exp(mean(logR0_D, na.rm=T) - Z*SElogR0_D) 
    # upper_D = exp(mean(logR0_D, na.rm=T) + Z*SElogR0_D) 
    # lower_E = exp(mean(logR0_E, na.rm=T) - Z*SElogR0_E) 
    # upper_E = exp(mean(logR0_E, na.rm=T) + Z*SElogR0_E) 
    # lower_F = exp(mean(logR0_F, na.rm=T) - Z*SElogR0_F) 
    # upper_F = exp(mean(logR0_F, na.rm=T) + Z*SElogR0_F) 
    # lower_G = exp(mean(logR0_G, na.rm=T) - Z*SElogR0_G) 
    # upper_G = exp(mean(logR0_G, na.rm=T) + Z*SElogR0_G) 
    # 
    # #R0_A <- read.table(paste0("R0_", sim_index, ".tab"), header=T)
    # #cluster_dynamics$mean_R0[1] = as.numeric(R0_A[1]) # Can't use exact R0 because standard deviation includes the high R0 of E and 0.
    # cluster_dynamics$mean_R0[1] = sum_sim$R0$R0.mean
    # cluster_dynamics$upper_R0[1] = quantile(sum_sim$R0$R0.dist, 0.95)
    # cluster_dynamics$lower_R0[1] = quantile(sum_sim$R0$R0.dist, 0.05)
    # 
    # cluster_dynamics$mean_R0[2] = mean(R0_B)
    # cluster_dynamics$upper_R0[2] = upper_B
    # cluster_dynamics$lower_R0[2] = lower_B
    # 
    # cluster_dynamics$mean_R0[3] = mean(R0_C)
    # cluster_dynamics$upper_R0[3] = upper_C
    # cluster_dynamics$lower_R0[3] = lower_C
    # 
    # cluster_dynamics$mean_R0[4] = mean(R0_D)
    # cluster_dynamics$upper_R0[4] = upper_D
    # cluster_dynamics$lower_R0[4] = lower_D
    # 
    # cluster_dynamics$mean_R0[5] = mean(R0_E)
    # cluster_dynamics$upper_R0[5] = upper_E
    # cluster_dynamics$lower_R0[5] = lower_E
    # 
    # cluster_dynamics$mean_R0[6] = mean(R0_F)
    # cluster_dynamics$upper_R0[6] = upper_F
    # cluster_dynamics$lower_R0[6] = lower_F
    # 
    # cluster_dynamics$mean_R0[7] = mean(R0_F)
    # cluster_dynamics$upper_R0[7] = upper_F
    # cluster_dynamics$lower_R0[7] = lower_F
    # 
    # cluster_dynamics$mean_R0[8] = mean(R0_G)
    # cluster_dynamics$upper_R0[8] = upper_G
    # cluster_dynamics$lower_R0[8] = lower_G
    # 
    # cluster_dynamics$mean_R0[8] = mean(R0_G)
    # cluster_dynamics$upper_R0[8] = upper_G
    # cluster_dynamics$lower_R0[8] = lower_G
    
    
    return(cluster_dynamics)
    
  }# End function
  cluster_dynamics <- true.cluster.dyn()
  
  write.csv(cluster_dynamics, paste0("data_", sim_index, ".csv"), 
              quote=F, row.names = F)
  
}  


## If any cluster exceeds background population in number, do not output results; however, if not, proceed with tree extraction.
max.t <- max(dynamics.table$t)
if(isTRUE(all(dynamics.table$Count[dynamics.table$state == 'A' & 
                                   dynamics.table$t == max.t] > 
              dynamics.table$Count[dynamics.table$state %in% c('B', 'C', 'D', 'E', 'F.1', 'F.2', 'G.1', 'G.2') & 
                                   dynamics.table$t == max.t]))) {
  save.data()
} else {NULL}
## End ###################################################################################

