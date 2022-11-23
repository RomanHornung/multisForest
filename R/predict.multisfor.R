#' Predict Using a Fitted Multi-Source Random Forest
#'
#' Obtain class probability predictions using a multi-source random forest
#' fitted using \code{\link{multisfor}}.
#'
#' @param object An object of class 'multisfor'. See \code{\link{multisfor}}.
#' @param data A \code{data.frame} containing data for which predictions should
#' be obtained. Must not contain missing values. Does not have to feature
#' all variables used to train the multi-source forest, but a subset of these.
#' @param weighted Should the predictions of the forests be weighted by their
#' OOB-AUC values (\code{TRUE}) before averaging them or should an un-weighted average 
#' be obtained (\code{FALSE}). Default is \code{TRUE}.
#' @return A vector of predicted probabilities for class '1'.
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
#' # Load the data (not included in the package currently)
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
#' 
#' # Get predicted class-probabilites for each obs. in the test-set:
#' 
#' predictions <- predict(object   = fw_rfs, 
#'                        data     = datatest,
#'                        weighted = TRUE)
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
predict.multisfor <- function(object, data, weighted = TRUE) {
  " Get predicitons for the 'data' from a fold-wise fitted RF ('object').
    To do so, process the test-data for each foldwise-fitted RF such that it has 
    the exact same lay-out as the data the fold-wise RF has been trained on.
    After that, prune each of the fold-wise fitted RFs in regard to the test-set
    (cut their nodes, if they use a non-availabe split-variable). Then each 
    fold-wise RF predicts the class probability for each obs. in the test-set. 
    
    Args:
      > object     (list): An object of class 'multisfor'
                           (see 'multisfor()'-function)
      > data (data.frame): Data we want to get predicitons for from our
                           FW_RFs - must not contain missing values & all
                           observations have to be observed in the same 
                           features. If the data doesn't contain features the
                           FW_RFs have been trained, no prediciton is possible
      > weighted   (bool): Shall the oob-AUC of the pruned fold-wise fitted trees
                           be used to create a weighted average of the prediciton?
                            -> else it will be an unweighted average
                                
    Return:
      > Vector with the predicted probability for each observation to belong to 
        class '1'.
  "
  # [0] Check inputs
  # 0-1 'object' has to be of class 'multisfor'
  assert_list(object)
  if (class(object) != 'multisfor') {
    stop("'object' must only contain objects of class 'multisfor'")
  }
  
  # 0-2 'data' has to be a data-frame w/o any missing values & min 1 obs.
  #        --> All obs. must be observed in the same features
  assert_data_frame(data, any.missing = F, min.rows = 1)
  
  # 0-3 'weighted' has to be a boolean
  assert_flag(weighted)
  
  # [1] Process 'data' (specific preparation for each FW-fitted RF)
  #     -> Convert 'data' to the same format of the data the FW-RFs were 
  #        originally trained with, to ensure factor levels/ features are the same ....
  tree_testsets <- list()
  for (i in 1:length(object)) {
    tree_testsets[[i]] <- process_test_data(tree      = object[[i]][[1]], 
                                            test_data = data)
  }
  
  # [2] Get a prediction for every observation in 'data' from each FW-RFs
  #     (class & probabilities) - as the features in Train & Test can be different, 
  #     the FW-fitted Forests need to be pruned before creating a prediction
  tree_preds_all <- list()
  tree_preds_all <- foreach(i = 1:length(object)) %do% { 
    
    # save the predictions as 'treeX_pred'
    get_pruned_prediction(trees    = object[[i]], 
                          test_set = tree_testsets[[i]])
  }
  
  # [3] Generate a (weighted) average of the predicitons
  # 3-1 Check whether any of the RFs is not usable [only NA predicitons] & rm it
  not_usable <- sapply(seq_len(length(tree_preds_all)), function(i) {
    all(is.na(tree_preds_all[[i]]$Class))
  })
  
  # 3-2 Check if the any of the trees are still usable after the pruning.
  #     And remove these trees that can not be used for predictions from 
  #     'tree_preds_all'
  if (all(not_usable)) {
    stop("None of the foldwise fitted RFs are usable for predictions!")
  } else if (any(not_usable)) {
    object         <- object[-c(which(not_usable))]
    tree_preds_all <- tree_preds_all[-c(which(not_usable))]
    tree_testsets  <- tree_testsets[-c(which(not_usable))]
  }
  
  # 3-3 Get the oob-metric for the remaining (& already pruned) FW-RFs
  # --3-1 Loop over all trees and prune them according to the testdata!
  for (i_ in 1:length(object)) {
    curr_test_set <- tree_testsets[[i_]]
    tmp           <- sapply(object[[i_]], FUN = function(x) x$prune(curr_test_set))
  }
  
  # --3-2 Get the oob-performance of the pruned trees!
  AUC_weight <- foreach(l = seq_len(length(object))) %do% { # par
    get_oob_AUC(trees = object[[l]])
  }
  
  # --3-3 Get the predicted probabilities for class 1 for each observation & 
  #       from each of the remaining FW-RFs
  probs_class_1_ <- sapply(1:nrow(data), FUN = function(x) {
    
    # Get a probability prediciton from each [still usable] tree!
    preds_all <- sapply(seq_len(length(tree_preds_all)), function(i) {
      tree_preds_all[[i]]$Probs[[x]][1]
    })
    
    # Combine the preditions of the different trees!
    if (weighted) {
      preds_all_w <- weighted.mean(preds_all, w = unlist(AUC_weight), na.rm = TRUE)
    } else {
      preds_all_w <- mean(preds_all, na.rm = TRUE)
    }
    
    preds_all_w
  })
  
  # [3] Return the predicted probabilities for each observation to belong 
  #     to class '1' 
  return(probs_class_1_)
}
