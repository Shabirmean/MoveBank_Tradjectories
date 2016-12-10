
library(ggplot2)
library(readr)
library(FactoMineR)
library(PCAmixdata)



eco <- read_csv("C:/Users/jerem/Google Drive/Education/McGill_/COMP551/ProjectR/ecology.csv", 
                col_types = cols(timestamp = col_datetime(format = "%Y-%m-%d %H:%M:%S")))

reduced <- eco[,c(23)]
meansMonth <- eco
meansMonth$month <-  format(eco$timestamp, "%b")

meansMonth <- as.data.frame(meansMonth)
names(meansMonth)[names(meansMonth)=="height-above-ellipsoid"] <- "height"
names(meansMonth)[names(meansMonth)=="location-long"] <- "long"
names(meansMonth)[names(meansMonth)=="location-lat"] <- "lati"
names(meansMonth)[names(meansMonth)=="individual-local-identifier"] <- "name"

altim <- aggregate( height ~ month+name, meansMonth, mean )
altsd <- aggregate( height ~ month+name, meansMonth, sd )
longi <- aggregate( long ~ name, meansMonth, mean )
lati <- aggregate( lati ~ name, meansMonth, mean )

plot(longi[[2]], lati[[2]])

unireduced <- reduced[!duplicated(reduced), ]


for (m in unique(format(eco$timestamp, "%b"))) {
  print(m)
  for(i in 1:nrow(unireduced)) {
    name <- unireduced[[i,1]]
    height <- altim[altim$month == m & altim$name ==name,]$height
    sd <- altsd[altsd$month == m & altsd$name ==name,]$height
    lo <- longi[longi$name == name,]$long
    if(lo < -91){
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 1
    }else if(lo < -90){
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 2
    }else{
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 3
    }
    
    if(length(height)>0){
      unireduced[unireduced$`individual-local-identifier` == name,m] = height
    }
    #if(length(sd)>0){
    #  unireduced[unireduced$`individual-local-identifier` == name,paste(m,'sd')] = sd
    #}

  }
  #unireduced[m] <- as.numeric(unlist(unireduced[m]))
}
names(unireduced)[names(unireduced)=="individual-local-identifier"] <- "name"
unireduced = na.omit(unireduced)
unireduced <- as.data.frame(unireduced)

unireduced[,'island'] <- as.factor(unireduced[,'island'] )

scaled <- unireduced
scaled[,-c(1,2)] <- t(scale(t(scaled[,-c(1,2)])))

#res <- PCA(scaled[,-c(1)])
res <- PCA(scaled[,-c(1,2)])
plot(res, title='', choice='ind',col.ind=scaled[,c(2)], label='var')
plot(res, title='', choice='varcor')

res.hcpc = HCPC(res)

