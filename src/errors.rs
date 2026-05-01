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