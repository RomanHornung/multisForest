#' Train a Multi-Source Random Forest Prediction Rule for Prediction using Partly Missing Multi-Omics Data
#'
#' Trains a multi-source random forest using multi-omics data that consists of
#' several subsets which differ according to their missingness patterns.
#' For example, if their are three data types A, B, C, the first subset may
#' feature data types A and C, the second subset B and C, the third subset
#' only A, and the fourth subset A, B, and C.
#'
#' @param data A \code{data.frame} containing the variables in the model.
#' Must contain a factor-valued column named 'ytarget' with the following
#' categories '1' (positive class) and '0' (negative class).
#' @param folds Vector of integers. Indicating for each row in 'data' to which 
#' subset it belongs. Observations in the same subset need to feature the same observed 
#' covariates.
#' @param num_trees Number of trees used to construct each forest.
#' @param mtry Number of variables to possibly split at in each node. Default is the (rounded down) square root of the number variables. 
#' @param min_node_size Minimal node size. Default is 1.
#' @return Object of class \code{multisfor}.
#' @examples
#' \donttest{
#'
#' # Load some necessary packages:
#' 
#' library(checkmate)
#' library(randomForestSRC)
#' library(caret)
#' library(pROC)
#' library(doParallel)
#' library(parallel)
#' library(ROCR)
#' library(caret)
#' 
#' 
#' # Load the data (not included in the package currently):
#' 
#' load("ExampleData.Rda")
#'
#'
#' # Fit a multi-source random forest:
#' 
#' fw_rfs <- multisfor(data          = datatrain, 
#'                     folds         = foldstrain,
#'                     num_trees     = 10, 
#'                     mtry          = 50,
#'                     min_node_size = 1)
#'
#' }
#'
#' @author Roman Hornung, Frederik Ludwigs
#' @references
#' \itemize{
#'   \item Breiman, L. (2001). Random forests. Mach Learn, 45:5-32, <\doi{10.1023/A:1010933404324}>.
#'   }
#' @encoding UTF-8
#' @export
multisfor <- function(data, folds, num_trees = 500, mtry = NULL, min_node_size = 1) {
  " Train a seperate RF on each fold in 'data', such that the final RF
    consists of as many FW-RFs as the data contains unique folds.
    
    Args:
      data (data.frame)  : Dataframe with dimensions n*p. Must contain a binary 
                           factor-column 'ytarget' (0 = neg. class / 1 = pos. class)
      folds (vec)        : Vector of length 'n' filled with integers. Indicating
                           for each row in 'data' to which fold it belongs.
                           ! Obs. in the same fold need to be observed in the 
                             same features !
      num_trees (int)    : Amount of trees that are used to grow per foldwise RF
      mtry (int)         : Amount of variables to be checked as split-variables
                           at every split. Default = ceiling(sqrt(p))
      min_node_size (int): Max. amount of nbservations that have to be in a 
                           terminal node!
                           
    Return: 
      List with as many fold-wise fitted RFs (of class 'multisourceRF')
      as the amount of unique folds in the data
  "
  # [0] Check inputs                                                        ----
  # 0-1 'data' has to be a data.frame & contain 'ytarget' as binary factor variable
  assert_data_frame(data)
  if (!('ytarget' %in% colnames(data))) {
    stop("'data' must contain 'ytarget' as column")
  } 
  assert_factor(data$ytarget, levels = c('0', '1'))
  
  # 0-2 'folds' must be of the same length as 'data' & must only contain integers
  if (nrow(data) != length(folds)) {
    stop("'folds' does not have the same length as 'data' has rows")
  }
  assert_integer(folds)
  
  # 0-3 Ensure that all obs. in the same fold have the same features
  for (curr_fold in unique(folds)) {

    # Get the observations of 'curr_fold'
    curr_obs <- which(folds == curr_fold)
    
    # Get the observed feature for the first observation in 'curr_obs'
    obs_feas_in_curr_fold <- colnames(data[curr_obs[1],
                                           which(!is.na(data[curr_obs[1],]), 
                                                 arr.ind=TRUE)])
    
    # Compare the observed features from the remaining 'curr_obs' to 
    # 'obs_feas_in_curr_fold' & ensure all feas are available for both
    check_common_colnames <- sapply(curr_obs[2:length(curr_obs)], function(x) {
      
      # observed columns for the current obs.
      colnames_curr_obs <- colnames(data[x, which(!is.na(data[x,]), 
                                                  arr.ind = TRUE)])
      
      check1 <- all(colnames_curr_obs %in% obs_feas_in_curr_fold)
      check2 <- all(obs_feas_in_curr_fold %in% colnames_curr_obs)
      
      any(!c(check1, check2))
    })
    
    if (any(check_common_colnames)) {
      stop("Observations in the same folds, do not have the same observed features!")
    }
  }
  
  # 0-4 'num_trees' & 'min_node_size' must be integers >= 1
  assert_int(num_trees, lower = 1)
  assert_int(min_node_size, lower = 1)
  
  # 0-5 'mtry' must be an int <= amount of cols in data - if not 'NULL'
  if (is.null(mtry)) {
    mtry = as.integer(ceiling(sqrt(ncol(data))))
  } else {
    assert_int(mtry, upper = ncol(data), lower = 1)
  }
  
  # [1] Fit a a separate RF on each of the available folds in the data      ----
  # 1-1 Initialize a list to store the fold-wise fitted RFs
  Forest <- list()
  
  # 1-2 Loop over each fold & fit a random-forest to each of them
  for (j_ in unique(folds)) {
    
    # --1 Extract the data for fold 'j_' & remove all columns w/ NAs 
    curr_fold <- data[which(folds == j_),]
    curr_fold <- curr_fold[,-which(sapply(curr_fold, function(x) sum(is.na(x)) == nrow(curr_fold)))]
    
    # --2 Define formula
    formula_all <- as.formula(paste("ytarget ~ ."))
    
    # --3 Define settings for the current foldwise RF 
    #     (settings for the arguments are the same as in the 'rfsrc'-package)
    fold_RF <- simpleRF(formula           = formula_all, 
                        data              = curr_fold, 
                        num_trees         = num_trees,
                        mtry              = as.integer(mtry), 
                        min_node_size     = min_node_size,
                        replace           = TRUE) # always TRUE, as we need OOB!
    
    # --4 Grow the single trees of the just defined 'fold_RF' 
    fold_RF <- lapply(fold_RF, function(x) {
      x$grow(replace = TRUE)
      x
    })
    
    # --5 Ensure the trees are grown correctly 
    #    (e.g. might be that no split-vars were found, ...)
    fold_RF <- all_trees_grown_correctly(fold_RF)
    
    # --6 Add the grown 'fold_RF' to 'Forest'
    Forest[[j_]] <- fold_RF
    
  }
  
  class(Forest) <- "multisfor"
  
  # 1-3 Return the fold-wise fitted RF's
  return(Forest)
}
