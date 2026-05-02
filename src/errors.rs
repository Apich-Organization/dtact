/// Generates a structured error enum with cold-path optimization.
///
/// This macro generates the error enumeration and a set of "cold" helper
/// functions that wrap the error variants in `Err`. These helpers are
/// marked with `#[cold]` and `#[inline(never)]` to ensure that error
/// generation does not pollute the instruction cache of the hot path.
#[doc(hidden)]
#[macro_export]
macro_rules! dtact_error {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( { $( $(#[$field_meta:meta])* $field:ident : $ftype:ty ),* $(,)? } )? $( ( $( $(#[$tuple_meta:meta])* $tname:ident : $ttype:ty ),* $(,)? ) )?
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant $( { $( $(#[$field_meta])* $field : $ftype ),* } )? $( ( $( $(#[$tuple_meta])* $ttype ),* ) )?,
            )*
        }

        pastey::paste! {
            $(
                $(#[$variant_meta])*
                #[doc(hidden)]
                #[cold]
                #[track_caller]
                #[inline(never)]
                pub const fn [<cold_ $name:snake _ $variant:snake>]<T>(
                    $($($field : $ftype),*)?
                    $($($tname : $ttype),*)?
                ) -> core::result::Result<T, $name> {
                    core::result::Result::Err($name::$variant $( { $($field),* } )? $( ( $( $tname ),* ) )?)
                }
            )*
        }
    };
}

dtact_error! {
    /// Errors encountered during Dtact Runtime operation.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub enum DtactError {
        /// OS-level memory mapping failed (Linux).
        MmapFailed,
        /// OS-level memory allocation failed (Windows).
        VirtualAllocFailed,
        /// OS-level memory protection change failed (Linux).
        MprotectFailed,
        /// OS-level memory protection change failed (Windows).
        VirtualProtectFailed,
        /// An API function was called from a thread not managed as a fiber.
        OutsideFiberContext,
    }
}
