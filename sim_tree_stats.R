
## Initialize stable working environment and store time of initiation
rm(list=ls())


ptm <- proc.time()
# List of packages for session
.packages <-  c("optparse", "remotes", "phytools", "treeio", "dplyr",  
                "plyr", "tidyr", "tidytree", "data.table",
                "parallel", "stringr", "rlist",  "phylodyn",
                "ggtree", "ggplot2", "TreeTools", "geiger", "adephylo", 
                "inflection", "skygrowth") # May need to incorporate code for familyR (https://rdrr.io/github/emillykkejensen/familyR/src/R/get_children.R) i fno longer supported.
.github_packages <- c("emillykkejensen/familyR") # "mrc-ide/skygrowth"

# # Install CRAN packages (if not already installed)
# .inst <- .packages %in% installed.packages()
# if(length(.packages[!.inst]) > 0) install.packages(.packages[!.inst])
# .inst_github <- .packages %in% installed.packages()
# ## Install GitHub packages(if not already installed)
# if(length(.github_packages[!.inst_github]) > 0) try(remotes::install_github(.github_packages[!.inst_github]))
# if(length(.github_packages[!.inst_github]) > 0) try(devtools::install_github(.github_packages[!.inst_github]))

# Load packages into session 
lapply(.packages, require, character.only=TRUE)
lapply(gsub(".+\\/(.+)", "\\1", .github_packages), require, character.only=TRUE)
numCores <- detectCores()

option_list = list(
  make_option(c("-s", "--sim_index"), type="numeric", default=1)
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


if (is.null(opt$s)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

print(opt)

### Functions ##############################################################################################################

`%notin%` <- Negate(`%in%`) # Just plain useful
`%!=na%` <- function(e1, e2) (is.na(e1) & is.na(e2)) # Also useful for finding nas in two dataframes


tree = list.files(pattern=paste0("sim_", opt$sim_index, "_.+\\.tree$"))
print("reading in tree...")
tree = lapply(tree, read.beast)[[1]]
tree.tbl <- as_tibble(tree)


clusters_data <- list.files(pattern=paste0("trans_clusters_", opt$sim_index, ".rds"))
print("reading in clusters...")
clusters_data = lapply(clusters_data, readRDS)[[1]]

cluster_dynamics = list.files(pattern=paste0("data_", opt$sim_index, ".csv"))
print("reading in tree...")
cluster_dynamics = lapply(cluster_dynamics, read.csv)[[1]]
c_order <- names(clusters_data)

cluster_dynamics <- cluster_dynamics[match(c_order, cluster_dynamics$cluster_id),]

clusters_phylo <- mclapply(clusters_data, function(x) {
  extract.clade(tree@phylo, findMRCA(tree@phylo, x$label))
}, mc.cores=numCores)

clusters_tbl <- mclapply(clusters_phylo, as_tibble, mc.cores=numCores)

#cluster_dynamics$v_mrsd <- as.Date(NA)
cluster_dynamics$v_timespan <- as.numeric(NA)
#cluster_dynamics$v_tmrca <- as.Date(NA)
#cluster_dynamics$v_days_before_end <- as.numeric(NA)

for (i in seq_along(clusters_data)) {
  mrsd <- as.Date(max(gsub(".+\\|(.+)", "\\1", tree@phylo$tip.label)))
  #cluster_dynamics$v_mrsd[i] <- max(clusters_data[[i]]$date, na.rm=T)
  cluster_dynamics$v_timespan[i] <- max(nodeHeights(clusters_phylo[[i]]))
  #cluster_dynamics$v_tmrca[i] <- cluster_dynamics$v_mrsd[i]-cluster_dynamics$v_timespan[i]
  #cluster_dynamics$v_days_before_end[i] <- difftime(mrsd, cluster_dynamics$v_tmrca[i],  units="days")
}



calculateNe <- function(tree) {
  tree <- multi2di(tree)
  res=round(max(nodeHeights(tree))/2) # Daily
  fit <- tryCatch(skygrowth.map(tree, res=res, tau0=0.1), error=function(e) NULL)
  p <- data.frame(time=max(nodeHeights(tree))-fit$time, nemed=fit$ne, ne_ci=fit$ne_ci)
  return(p)
}
calculateR0 <- function(Ne, conf.level=0.95) {
  s <- 100 # sample size of 100
#  psi <- rnorm(s, 0.038, 0.014) #Duration of infection around 14 days
  psi <- rnorm(s, 14, 5) #Duration of infection around 14 days -incubation period of 5 days
#  psi <- rnorm(s, 9, 5) #Duration of infection around 14 days -incubation period of 5 days
  Z=qnorm(0.5*(1 + conf.level))
  Re <- list()
  if (isTRUE(nrow(Ne) >1)) {
    for (i in 2:nrow(Ne)) {
      time = Ne$time[i-1]
      Re.dist = sample(1+psi*(Ne$nemed[i]-Ne$nemed[i-1])/((Ne$time[i]-Ne$time[i-1])*Ne$nemed[i-1]),
                       size=s, replace=F)
      logRe = log(Re.dist)
      SElogRe = sd(logRe)/sqrt(s) # standard deviation of 2, sample size of 10
      LCL = exp(mean(logRe) - Z*SElogRe)
      UCL = exp(mean(logRe) + Z*SElogRe)
      Re[[i-1]] <- data.frame(time = time, mean_Re=mean(Re.dist), conf.int=paste0("(", LCL, "," ,UCL, ")"))
    }
  } else {Re <- list(NULL)}
  Re <- do.call("rbind",Re)
  R0 <- Re$mean_Re[1]
  return(R0)
}
calculateLTT <- function(tree) {
    fit <- ltt(tree, plot=F)
    df <- distinct(data.frame(ne=fit$ltt, time=fit$times))
    df <- df[-1,]
    df <- data.frame(x=df$time, y=df$ne)
  
  repeat {
    i <- 2
    while (i <= nrow(df)) {
      if (isTRUE(df$x[i]==df$x[i-1])){
        min_y <- min(df$y[i-1],df$y[i])
        unwanted <- df[df$y== min_y & (df$x==df$x[i] | df$x==df$x[i-1]),]
      } else{ unwanted <- data.frame(x=NA, y=NA)}
      df <- setdiff(df, unwanted)
      i <- i+1
    }
    if(length(unique(df$x))==length(df$x)) {
      break
    }
  }
  
   
  cc <- check_curve(df$x, log(df$y))$ctype
  # if (cc=="convex") {
  #   model="constant"
  # } else {
  #   if (cc=="concave") {
  #     model="growth"
  #   } else { model="mixed"}
  # }
  result <- data.frame()
return(cc)
}
calculateGrowth <- function(cluster_Ne, total_r) {# modelNe_log <- function(Ne_df) {
  cluster_df <- data.frame(x=cluster_Ne$time, y=cluster_Ne$nemed)
  if (nrow(cluster_df) > 1) {
    max_cluster_ne <- max(cluster_df$y, na.rm=T); min_cluster_ne <- min(cluster_df$y, na.rm=T)
    max_cluster_t <- max(cluster_df$x, na.rm=T); min_cluster_t <- min(cluster_df$x, na.rm=T)
    cluster_r = (max_cluster_ne-min_cluster_ne)/(max_cluster_t-min_cluster_t)
  
    relative_r <- cluster_r/total_r
  
    rmax <- list()
    for (i in 2:nrow(cluster_df)) {
     rmax[[i]] <- (cluster_df$y[i]-cluster_df$y[i-1])/(cluster_df$x[i]-cluster_df$x[i-1])
    }
    rmax <- max(do.call("rbind",rmax), na.rm=T)
  
  #Fraction of time spent in growth
    max_cluster_t.ne <- cluster_df$x[cluster_df$y==max_cluster_ne]
    t_frac <- (max_cluster_t.ne-min_cluster_t)/(max_cluster_t-min_cluster_t)
  } else {
    cluster_r <- NA
    relative_r <- NA
    t_frac <- NA
    rmax <- NA
  }
  
  result <- data.frame(abs_growth_rate=cluster_r, rel_growth_rate=relative_r, t_frac=t_frac, rmax=rmax)
  return(result)
}
calculateGamma <- function(tree) {
  tree <- multi2di(tree)
  gamma <- gammaStat(tree)
  return(gamma)
}
calculateOster <- function(tree) {
  tree <- multi2di(tree)  
  cluster_size <- length(tree$tip.label)-1
  sum_heights <- sum(nodeHeights(tree))
  longest <- max(nodeHeights(tree))
  co <- cluster_size/sum_heights + longest
  return(co)
}
timeData <- function(tree, clusters) {
  sts <- distRoot(tree@phylo, tips = "all", method="patristic")
  sts <- data.frame(time = sts, ID=names(sts))
  assign("sts", setNames(sts$time, sts$ID), envir = globalenv())

  tbl_list <- mclapply(clusters, function(x) {
    mrsd <- max(sts$time[sts$ID %in% x$tip.label], na.rm=T)
    cluster_size <- length(x$tip.label)
    root_age <- mrsd-max(nodeHeights(x))
    x <- cbind(as_tibble(x), mrsd = mrsd,
               cluster_size = cluster_size,
               root_age = root_age 
    )}, mc.cores=numCores) # End creation of tbl_list
  
  assign("tbl_list", tbl_list, envir = globalenv())
  # clust_metadata <- rbindlist(tbl_list, fill=T) %>%
  #   dplyr::select(-parent, -node, -branch.length) # No longer need parenta and node information, since not unique
  # names(clust_metadata)[1] <- "ID"
  
  # sts_list <- mclapply(clusters, function(c) {
  #   c <- sts[sts$ID %in% c$tip.label,]
  #   c <- setNames(c$time, c$ID)
  # }, mc.cores=numCores)
  # 
  # return(sts_list)
}
calculateNeeBD <- function(tree) {
  tree <- multi2di(tree)
  #tree = prune.extinct.taxa(tree)
  fit.bd <- tryCatch(birthdeath(tree), error=function(e) NULL)
  if (!is.null(fit.bd)) {
    b <- bd(fit.bd)[1]
  } else {
    b <- NULL
  }
  return(b)
}
calculatePD <- function(tree) {
  tree <- multi2di(tree)
  pd <- sum(tree$edge.length)
  return(pd)
}
calculateCC <- function(tree) {
  tree <- multi2di(tree)
  out <- capture.output(
    tryCatch(cherry(tree), error=function(e) NULL)
  )
  ntips <- as.numeric(gsub("Number of tips\\: ([0-9]+) ", "\\1", out[5]))
  ncherries <- as.numeric(gsub("Number of cherries\\: ([0-9]+) ", "\\1", out[6]))
  cpn <- ncherries/ntips
  return(cpn)
}

################################################################################################

total_Ne <- calculateNe(tree@phylo)
total_gamma <- calculateGamma(tree@phylo)
total_oster <- calculateOster(tree@phylo)
total_NeeBD <- calculateNeeBD(tree@phylo)
total_PD <- calculatePD(tree@phylo)

total_df <- data.frame(x=total_Ne$time, y=total_Ne$nemed)
max_total_ne <- max(total_df$y); min_total_ne <- min(total_df$y)
max_total_t <- total_df$x[total_df$y==max_total_ne]; min_total_t <- min(total_df$x)
total_r = (max_total_ne-min_total_ne)/(max_total_t-min_total_t)

cluster_Ne <- mclapply(clusters_phylo, calculateNe, mc.cores=numCores)
cluster_R0 <- mclapply(cluster_Ne, calculateR0, conf.level=0.95, mc.cores=numCores)
cluster_gamma <- mclapply(clusters_phylo, calculateGamma, mc.cores=numCores)
cluster_oster <- mclapply(clusters_phylo, calculateOster, mc.cores=numCores)
cluster_growth_rates <- mclapply(cluster_Ne, function(x) {
  calculateGrowth(x, total_r)}, mc.cores=numCores)
cluster_ltt_shape <- mclapply(clusters_phylo, calculateLTT, mc.cores=numCores)
cluster_cherries <- mclapply(clusters_phylo, calculateCC, mc.cores=numCores)
cluster_NeeBD <- mclapply(clusters_phylo, calculateNeeBD, mc.cores=numCores)
cluster_PD <- mclapply(clusters_phylo, calculatePD, mc.cores=numCores)

## Populate cluster_dynamics table
cluster_dynamics$gamma <- as.numeric(NA)
cluster_dynamics$oster <- as.numeric(NA)
cluster_dynamics$birth_rate <- as.numeric(NA)
#cluster_dynamics$rel_birth_rate <- as.numeric(NA)
cluster_dynamics$PD <- as.numeric(NA)
#cluster_dynamics$rel_PD <- as.numeric(NA)
cluster_dynamics$R0 <- as.numeric(NA)
cluster_dynamics$abs_growth_rate <- as.numeric(NA)
#cluster_dynamics$rel_growth_rate <- as.numeric(NA)
cluster_dynamics$fraction_time_growth <- as.numeric(NA)
cluster_dynamics$r_max <- as.numeric(NA)
cluster_dynamics$ltt_shape <- as.character(NA)
cluster_dynamics$cherries <- as.character(NA)

for (i in 1:nrow(cluster_dynamics)) {
  cluster_dynamics$gamma[i] <- cluster_gamma[[i]]
  cluster_dynamics$oster[i] <- cluster_oster[[i]]
  cluster_dynamics$birth_rate[i] <- cluster_NeeBD[[i]]
  #cluster_dynamics$rel_birth_rate[i] <- cluster_NeeBD[[i]]/total_NeeBD
  cluster_dynamics$PD[i] <- cluster_PD[[i]]
  #cluster_dynamics$rel_PD[i] <- cluster_PD[[i]]/total_PD
  cluster_dynamics$R0[i] <- cluster_R0[[i]]
  cluster_dynamics$abs_growth_rate[i] <- cluster_growth_rates[[i]]$abs_growth_rate
  #cluster_dynamics$rel_growth_rate[i] <- cluster_growth_rates[[i]]$rel_growth_rate
  cluster_dynamics$fraction_time_growth[i] <- cluster_growth_rates[[i]]$t_frac
  cluster_dynamics$r_max[i] <- cluster_growth_rates[[i]]$rmax
  cluster_dynamics$ltt_shape[i] <- cluster_ltt_shape[[i]]
  cluster_dynamics$cherries[i] <- as.numeric(cluster_cherries[[i]])
}


final_df <- merge(cluster_dynamics, tree@data, by="cluster_id", all.y=T)

final_df$sim=opt$sim_index

write.csv(final_df, paste0("cluster_dynamics_", opt$sim_index, ".csv"), row.names = F, quote=F)

n <- length(tree@phylo$edge.length)
rand_bl <- runif(n, 8E-04, 0.001)
bl <- data.frame(from=tree@phylo$edge[,1], 
                 to=tree@phylo$edge[,2], 
                 weight1=tree@phylo$edge.length,
                 weight2=tree@phylo$edge.length*rand_bl,
                 sim=opt$sim_index)

write.csv(bl, paste0("edges_", opt$sim_index, ".csv"), row.names = F, quote=F)





# metadata <- data.frame(id=tree@phylo$tip.label, 
#                        date=gsub(".+\\|(.+)", "\\1", tree@phylo$tip.label))
## IQ-TREE changes names, so need to change here
# metadata$id <- gsub("\\|", "_", metadata$id)
# write.csv(metadata, "metadata.csv", quote=F, row.names = F)

